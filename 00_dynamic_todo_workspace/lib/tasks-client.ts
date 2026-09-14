export type TaskPriority = 'High' | 'Medium' | 'Low';

export type Task = {
  id: string;
  title: string;
  notes: string;
  project: string;
  priority: TaskPriority;
  dueDate: string | null;
  estimateMinutes: number;
  completed: boolean;
  createdAt: string;
  updatedAt: string;
};

export type TaskInput = {
  title: string;
  notes?: string;
  project?: string;
  priority?: TaskPriority;
  dueDate?: string | null;
  estimateMinutes?: number;
};

export type TaskUpdate = Partial<TaskInput> & { completed?: boolean };

async function jsonRequest<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set('Content-Type', 'application/json');

  const response = await fetch(url, {
    ...init,
    headers,
  });
  const body = (await response.json()) as T & { error?: string };
  if (!response.ok) throw new Error(body.error ?? 'The request failed.');
  return body;
}

export async function fetchTasks() {
  const result = await jsonRequest<{ tasks: Task[] }>('/api/tasks');
  return result.tasks;
}

export async function createTask(input: TaskInput) {
  const result = await jsonRequest<{ task: Task }>('/api/tasks', {
    method: 'POST',
    body: JSON.stringify(input),
  });
  return result.task;
}

export async function updateTask(id: string, input: TaskUpdate) {
  const result = await jsonRequest<{ task: Task }>(
    `/api/tasks/${encodeURIComponent(id)}`,
    { method: 'PATCH', body: JSON.stringify(input) },
  );
  return result.task;
}

export async function removeTask(id: string) {
  await jsonRequest<{ deleted: true }>(`/api/tasks/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
}
