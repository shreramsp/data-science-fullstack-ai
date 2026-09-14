import { createTask, listTasks, type TaskPriority } from '@/db/tasks';

const priorities = new Set<TaskPriority>(['High', 'Medium', 'Low']);

function parseCreateInput(value: unknown) {
  if (!value || typeof value !== 'object') {
    throw new Error('A task object is required.');
  }

  const input = value as Record<string, unknown>;
  const title = typeof input.title === 'string' ? input.title.trim() : '';
  if (!title || title.length > 160) {
    throw new Error('Title must contain 1–160 characters.');
  }

  const priority =
    typeof input.priority === 'string' && priorities.has(input.priority as TaskPriority)
      ? (input.priority as TaskPriority)
      : 'Medium';
  const estimateMinutes = Number(input.estimateMinutes ?? 25);

  return {
    title,
    notes: typeof input.notes === 'string' ? input.notes.slice(0, 2000) : '',
    project:
      typeof input.project === 'string' && input.project.trim()
        ? input.project.trim().slice(0, 80)
        : 'Inbox',
    priority,
    dueDate:
      typeof input.dueDate === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(input.dueDate)
        ? input.dueDate
        : null,
    estimateMinutes:
      Number.isFinite(estimateMinutes) && estimateMinutes >= 5 && estimateMinutes <= 480
        ? Math.round(estimateMinutes)
        : 25,
  };
}

export async function GET() {
  try {
    return Response.json({ tasks: await listTasks() });
  } catch (error) {
    console.error('Unable to list tasks', error);
    return Response.json({ error: 'Unable to load tasks.' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const task = await createTask(parseCreateInput(await request.json()));
    return Response.json({ task }, { status: 201 });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unable to create task.';
    const status = message.startsWith('Unable') ? 500 : 400;
    return Response.json({ error: message }, { status });
  }
}
