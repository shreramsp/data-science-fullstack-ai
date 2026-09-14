INSERT OR IGNORE INTO tasks
  (id, title, notes, project, priority, due_date, estimate_minutes, completed, created_at, updated_at)
VALUES
  ('demo-brief', 'Shape the product brief', 'Clarify the audience, outcome, and acceptance criteria before implementation.', 'Launch week', 'High', '2026-09-13', 25, 0, '2026-09-13T08:00:00.000Z', '2026-09-13T08:00:00.000Z'),
  ('demo-dashboard', 'Review dashboard interactions', 'Check keyboard navigation and the mobile task flow.', 'Orbit redesign', 'Medium', '2026-09-13', 40, 0, '2026-09-13T08:15:00.000Z', '2026-09-13T08:15:00.000Z'),
  ('demo-progress', 'Send weekly progress note', 'Summarize completed work and blockers for the team.', 'Operations', 'Low', '2026-09-13', 15, 0, '2026-09-13T08:30:00.000Z', '2026-09-13T08:30:00.000Z'),
  ('demo-research', 'Organize interview highlights', 'Group insights into themes for the research synthesis.', 'Research', 'Medium', '2026-09-13', 30, 1, '2026-09-13T08:45:00.000Z', '2026-09-13T09:00:00.000Z'),
  ('demo-critique', 'Prepare design critique notes', 'Capture the three decisions needed from Monday review.', 'Orbit redesign', 'High', '2026-09-14', 35, 0, '2026-09-13T09:15:00.000Z', '2026-09-13T09:15:00.000Z');
