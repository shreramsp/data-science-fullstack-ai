import {
  deleteTask,
  updateTask,
  type TaskPriority,
  type UpdateTaskInput,
} from '@/db/tasks';

const priorities = new Set<TaskPriority>(['High', 'Medium', 'Low']);

function parseUpdateInput(value: unknown): UpdateTaskInput {
  if (!value || typeof value !== 'object') {
    throw new Error('A task update object is required.');
  }

  const input = value as Record<string, unknown>;
  const update: UpdateTaskInput = {};

  if ('title' in input) {
    const title = typeof input.title === 'string' ? input.title.trim() : '';
    if (!title || title.length > 160) {
      throw new Error('Title must contain 1–160 characters.');
    }
    update.title = title;
  }
  if (typeof input.notes === 'string') update.notes = input.notes.slice(0, 2000);
  if (typeof input.project === 'string' && input.project.trim()) {
    update.project = input.project.trim().slice(0, 80);
  }
  if (typeof input.priority === 'string') {
    if (!priorities.has(input.priority as TaskPriority)) {
      throw new Error('Priority must be High, Medium, or Low.');
    }
    update.priority = input.priority as TaskPriority;
  }
  if (input.dueDate === null) update.dueDate = null;
  if (
    typeof input.dueDate === 'string' &&
    /^\d{4}-\d{2}-\d{2}$/.test(input.dueDate)
  ) {
    update.dueDate = input.dueDate;
  }
  if (typeof input.estimateMinutes === 'number') {
    if (input.estimateMinutes < 5 || input.estimateMinutes > 480) {
      throw new Error('Estimate must be between 5 and 480 minutes.');
    }
    update.estimateMinutes = Math.round(input.estimateMinutes);
  }
  if (typeof input.completed === 'boolean') update.completed = input.completed;

  return update;
}

type RouteContext = { params: Promise<{ id: string }> };

export async function PATCH(request: Request, context: RouteContext) {
  try {
    const { id } = await context.params;
    const task = await updateTask(id, parseUpdateInput(await request.json()));
    if (!task) return Response.json({ error: 'Task not found.' }, { status: 404 });
    return Response.json({ task });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unable to update task.';
    const status = message.startsWith('Unable') ? 500 : 400;
    return Response.json({ error: message }, { status });
  }
}

export async function DELETE(_request: Request, context: RouteContext) {
  try {
    const { id } = await context.params;
    const deleted = await deleteTask(id);
    if (!deleted) return Response.json({ error: 'Task not found.' }, { status: 404 });
    return Response.json({ deleted: true });
  } catch (error) {
    console.error('Unable to delete task', error);
    return Response.json({ error: 'Unable to delete task.' }, { status: 500 });
  }
}
