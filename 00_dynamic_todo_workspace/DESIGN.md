# Zenith Design and Architecture

## Product goal

Zenith helps an individual turn an unstructured list into a calm, actionable day. The most important interaction—adding a task—is available in the first viewport, while progress, prioritization, and focus tools remain visible without overwhelming the task list.

## Experience principles

1. **Work first:** the app opens on the task workspace rather than a promotional landing page.
2. **One clear next move:** strong hierarchy keeps task titles, priority, due date, and estimate scannable.
3. **Safe mutations:** editing is reviewable and deletion requires explicit confirmation.
4. **Honest feedback:** progress and counts are calculated from persisted records.
5. **Keyboard and mobile ready:** all controls have accessible names, search has a `/` shortcut, and primary views remain available on narrow screens.

## Visual direction

The interface uses a dark graphite navigation rail against a soft paper-colored workspace. Electric indigo carries progress and focus actions; citrus lime identifies moments of momentum. Rounded but restrained surfaces keep the product approachable while compact metadata preserves the density expected from a productivity tool.

## Architecture

```text
Browser UI
  ├── task views, search, sorting, layouts
  ├── task editor and delete confirmation
  ├── focus timer
  └── optional WebMCP create_task tool
          │
          ▼
Route handlers
  ├── GET/POST /api/tasks
  └── PATCH/DELETE /api/tasks/:id
          │
          ▼
Task repository
  ├── prepared statements
  ├── row-to-domain mapping
  └── validation at API boundary
          │
          ▼
D1 / SQLite
  ├── tasks table
  ├── completed + due-date index
  └── project index
```

## Data model

Each task stores a durable ID, title, notes, project, priority, optional due date, time estimate, completion state, and creation/update timestamps. The API validates user-controlled fields before executing parameterized queries.

Indexes match actual workspace access patterns: filtering open tasks by due date and grouping/filtering by project.

## State and failure handling

The database is the source of truth. The client loads records on entry and updates its local projection only after a successful API response. Loading skeletons prevent layout shifts, errors provide a retry path, and filtered collections use an explicit empty state.

## Accessibility

- Semantic headings, navigation regions, forms, articles, and progress output
- Accessible labels for icon-only and checkbox controls
- Visible keyboard focus supplied by the shared component primitives
- Minimum 14–16px text for working labels and body content
- Controls remain usable in horizontal mobile navigation
- Color is supplemented by text labels for priority, project, and state

## Security and integrity

- Parameterized D1 statements prevent SQL injection.
- Server-side length, enum, date, and estimate validation constrains input.
- Delete routes target one explicit task ID.
- Schema changes are migration-owned rather than created at request time.
- No authentication is included because Project 00 is designed as a single-user local application.
