'use client';

import { useCallback, useEffect, useState } from 'react';

import {
  createTask as createTaskRequest,
  fetchTasks,
  removeTask as removeTaskRequest,
  updateTask as updateTaskRequest,
  type Task,
  type TaskInput,
  type TaskUpdate,
} from '@/lib/tasks-client';

type ModelContext = {
  registerTool: (
    tool: {
      name: string;
      title: string;
      description: string;
      inputSchema: object;
      annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
      execute: (input: unknown) => Promise<unknown>;
    },
    options?: { signal?: AbortSignal },
  ) => void | Promise<void>;
};

declare global {
  interface Document {
    modelContext?: ModelContext;
  }
}

export function useTasks() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setTasks(await fetchTasks());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to load tasks.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;

    void fetchTasks()
      .then((items) => {
        if (!active) return;
        setTasks(items);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (!active) return;
        setError(cause instanceof Error ? cause.message : 'Unable to load tasks.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  const createTask = useCallback(async (input: TaskInput) => {
    const task = await createTaskRequest(input);
    setTasks((current) => [...current, task]);
    return task;
  }, []);

  const updateTask = useCallback(async (id: string, input: TaskUpdate) => {
    const task = await updateTaskRequest(id, input);
    setTasks((current) => current.map((item) => (item.id === id ? task : item)));
    return task;
  }, []);

  const removeTask = useCallback(async (id: string) => {
    await removeTaskRequest(id);
    setTasks((current) => current.filter((item) => item.id !== id));
  }, []);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;

    const lifecycle = new AbortController();
    const register = context.registerTool(
      {
        name: 'create_task',
        title: 'Create task',
        description:
          'Create a task in the visible Zenith workspace with an optional project, priority, due date, and time estimate.',
        inputSchema: {
          type: 'object',
          properties: {
            title: { type: 'string', minLength: 1, maxLength: 160 },
            project: { type: 'string', maxLength: 80 },
            priority: { type: 'string', enum: ['High', 'Medium', 'Low'] },
            dueDate: { type: 'string', pattern: '^\\d{4}-\\d{2}-\\d{2}$' },
            estimateMinutes: { type: 'integer', minimum: 5, maximum: 480 },
          },
          required: ['title'],
          additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        async execute(value) {
          if (!value || typeof value !== 'object') {
            throw new Error('A task object is required.');
          }
          const input = value as Record<string, unknown>;
          if (typeof input.title !== 'string' || !input.title.trim()) {
            throw new Error('A non-empty title is required.');
          }
          const task = await createTask({
            title: input.title,
            project: typeof input.project === 'string' ? input.project : 'Inbox',
            priority:
              input.priority === 'High' ||
              input.priority === 'Low' ||
              input.priority === 'Medium'
                ? input.priority
                : 'Medium',
            dueDate: typeof input.dueDate === 'string' ? input.dueDate : null,
            estimateMinutes:
              typeof input.estimateMinutes === 'number' ? input.estimateMinutes : 25,
          });
          return { id: task.id, title: task.title, status: 'created' };
        },
      },
      { signal: lifecycle.signal },
    );

    void Promise.resolve(register).catch(() => {
      // Unsupported or experimental browsers may reject registration.
    });
    return () => lifecycle.abort();
  }, [createTask]);

  return { tasks, loading, error, load, createTask, updateTask, removeTask };
}
