import { env } from 'cloudflare:workers';

export type TaskPriority = 'High' | 'Medium' | 'Low';

export type TaskRecord = {
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

type TaskRow = {
  id: string;
  title: string;
  notes: string;
  project: string;
  priority: TaskPriority;
  due_date: string | null;
  estimate_minutes: number;
  completed: number;
  created_at: string;
  updated_at: string;
};

export type CreateTaskInput = {
  title: string;
  notes?: string;
  project?: string;
  priority?: TaskPriority;
  dueDate?: string | null;
  estimateMinutes?: number;
};

export type UpdateTaskInput = Partial<CreateTaskInput> & {
  completed?: boolean;
};

function database() {
  if (!env.DB) throw new Error('Task database is not available.');
  return env.DB;
}

function mapTask(row: TaskRow): TaskRecord {
  return {
    id: row.id,
    title: row.title,
    notes: row.notes,
    project: row.project,
    priority: row.priority,
    dueDate: row.due_date,
    estimateMinutes: row.estimate_minutes,
    completed: Boolean(row.completed),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

export async function listTasks(): Promise<TaskRecord[]> {
  const result = await database()
    .prepare(
      `SELECT id, title, notes, project, priority, due_date, estimate_minutes,
              completed, created_at, updated_at
       FROM tasks
       ORDER BY completed ASC,
                CASE priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
                due_date ASC,
                created_at ASC`,
    )
    .all<TaskRow>();

  return result.results.map(mapTask);
}

export async function findTask(id: string): Promise<TaskRecord | null> {
  const row = await database()
    .prepare(
      `SELECT id, title, notes, project, priority, due_date, estimate_minutes,
              completed, created_at, updated_at
       FROM tasks
       WHERE id = ?`,
    )
    .bind(id)
    .first<TaskRow>();

  return row ? mapTask(row) : null;
}

export async function createTask(input: CreateTaskInput): Promise<TaskRecord> {
  const id = crypto.randomUUID();
  const now = new Date().toISOString();

  await database()
    .prepare(
      `INSERT INTO tasks
        (id, title, notes, project, priority, due_date, estimate_minutes,
         completed, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)`,
    )
    .bind(
      id,
      input.title,
      input.notes ?? '',
      input.project ?? 'Inbox',
      input.priority ?? 'Medium',
      input.dueDate ?? null,
      input.estimateMinutes ?? 25,
      now,
      now,
    )
    .run();

  const task = await findTask(id);
  if (!task) throw new Error('Task was created but could not be read.');
  return task;
}

export async function updateTask(
  id: string,
  input: UpdateTaskInput,
): Promise<TaskRecord | null> {
  const current = await findTask(id);
  if (!current) return null;

  const next = {
    title: input.title ?? current.title,
    notes: input.notes ?? current.notes,
    project: input.project ?? current.project,
    priority: input.priority ?? current.priority,
    dueDate: input.dueDate === undefined ? current.dueDate : input.dueDate,
    estimateMinutes: input.estimateMinutes ?? current.estimateMinutes,
    completed: input.completed ?? current.completed,
  };

  await database()
    .prepare(
      `UPDATE tasks
       SET title = ?, notes = ?, project = ?, priority = ?, due_date = ?,
           estimate_minutes = ?, completed = ?, updated_at = ?
       WHERE id = ?`,
    )
    .bind(
      next.title,
      next.notes,
      next.project,
      next.priority,
      next.dueDate,
      next.estimateMinutes,
      next.completed ? 1 : 0,
      new Date().toISOString(),
      id,
    )
    .run();

  return findTask(id);
}

export async function deleteTask(id: string): Promise<boolean> {
  const result = await database()
    .prepare('DELETE FROM tasks WHERE id = ?')
    .bind(id)
    .run();

  return (result.meta.changes ?? 0) > 0;
}
