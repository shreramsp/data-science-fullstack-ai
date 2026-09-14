const baseUrl = process.env.ZENITH_BASE_URL ?? 'http://localhost:3000';
let createdId;

async function request(path, init) {
  const response = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(`${init?.method ?? 'GET'} ${path} failed: ${JSON.stringify(body)}`);
  }
  return body;
}

try {
  const initial = await request('/api/tasks');
  if (!Array.isArray(initial.tasks)) throw new Error('Task list is not an array.');

  const created = await request('/api/tasks', {
    method: 'POST',
    body: JSON.stringify({
      title: 'Automated smoke-test task',
      project: 'Inbox',
      priority: 'Low',
      estimateMinutes: 5,
    }),
  });
  createdId = created.task.id;

  const updated = await request(`/api/tasks/${encodeURIComponent(createdId)}`, {
    method: 'PATCH',
    body: JSON.stringify({ title: 'Smoke test verified', completed: true }),
  });
  if (!updated.task.completed || updated.task.title !== 'Smoke test verified') {
    throw new Error('Updated task did not match the requested state.');
  }

  await request(`/api/tasks/${encodeURIComponent(createdId)}`, { method: 'DELETE' });
  createdId = undefined;
  console.log(`Smoke test passed against ${baseUrl}.`);
} finally {
  if (createdId) {
    await request(`/api/tasks/${encodeURIComponent(createdId)}`, {
      method: 'DELETE',
    }).catch(() => undefined);
  }
}
