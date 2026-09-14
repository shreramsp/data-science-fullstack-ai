import { index, integer, sqliteTable, text } from 'drizzle-orm/sqlite-core';

export const tasks = sqliteTable(
  'tasks',
  {
    id: text('id').primaryKey(),
    title: text('title').notNull(),
    notes: text('notes').notNull().default(''),
    project: text('project').notNull().default('Inbox'),
    priority: text('priority', { enum: ['High', 'Medium', 'Low'] })
      .notNull()
      .default('Medium'),
    dueDate: text('due_date'),
    estimateMinutes: integer('estimate_minutes').notNull().default(25),
    completed: integer('completed', { mode: 'boolean' })
      .notNull()
      .default(false),
    createdAt: text('created_at').notNull(),
    updatedAt: text('updated_at').notNull(),
  },
  (table) => [
    index('idx_tasks_completed_due').on(table.completed, table.dueDate),
    index('idx_tasks_project').on(table.project),
  ],
);
