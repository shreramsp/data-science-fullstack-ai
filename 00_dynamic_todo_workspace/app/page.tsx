'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Bell,
  CalendarDays,
  Check,
  CheckCircle2,
  ChevronDown,
  Circle,
  Clock3,
  Inbox,
  LayoutGrid,
  List,
  ListTodo,
  MoreHorizontal,
  Pause,
  Pencil,
  Play,
  Plus,
  RotateCcw,
  Search,
  Settings2,
  Sparkles,
  SunMedium,
  TimerReset,
  Trash2,
} from 'lucide-react';

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from '@/components/ui/empty';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { Textarea } from '@/components/ui/textarea';
import { useTasks } from '@/hooks/use-tasks';
import type { Task, TaskPriority } from '@/lib/tasks-client';
import { cn } from '@/lib/utils';

type ViewKey = 'today' | 'inbox' | 'upcoming' | 'all' | 'completed' | `project:${string}`;
type SortKey = 'smart' | 'due' | 'title';
type LayoutMode = 'list' | 'board';

type TaskDraft = {
  title: string;
  notes: string;
  project: string;
  priority: TaskPriority;
  dueDate: string;
  estimateMinutes: string;
};

const projects = [
  { name: 'Launch week', color: 'bg-[#ff6b52]' },
  { name: 'Orbit redesign', color: 'bg-[#8b7cff]' },
  { name: 'Operations', color: 'bg-[#4bc6a6]' },
  { name: 'Research', color: 'bg-[#f3be4a]' },
];

const projectColors: Record<string, string> = Object.fromEntries(
  projects.map((project) => [project.name, project.color]),
);

function localDateKey(date = new Date()) {
  const offset = date.getTimezoneOffset() * 60_000;
  return new Date(date.getTime() - offset).toISOString().slice(0, 10);
}

function readableDate(date: string | null) {
  if (!date) return 'No date';
  const today = localDateKey();
  const tomorrow = localDateKey(new Date(Date.now() + 86_400_000));
  if (date === today) return 'Today';
  if (date === tomorrow) return 'Tomorrow';
  if (date < today) return 'Overdue';
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
  }).format(new Date(`${date}T12:00:00`));
}

function taskDraft(task: Task): TaskDraft {
  return {
    title: task.title,
    notes: task.notes,
    project: task.project,
    priority: task.priority,
    dueDate: task.dueDate ?? '',
    estimateMinutes: String(task.estimateMinutes),
  };
}

function formatTimer(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
}

export default function Home() {
  const { tasks, loading, error, load, createTask, updateTask, removeTask } =
    useTasks();
  const [draft, setDraft] = useState('');
  const [query, setQuery] = useState('');
  const [view, setView] = useState<ViewKey>('today');
  const [sort, setSort] = useState<SortKey>('smart');
  const [layout, setLayout] = useState<LayoutMode>('list');
  const [saving, setSaving] = useState(false);
  const [mutationError, setMutationError] = useState<string | null>(null);
  const [editing, setEditing] = useState<Task | null>(null);
  const [editDraft, setEditDraft] = useState<TaskDraft | null>(null);
  const [deleting, setDeleting] = useState<Task | null>(null);
  const [timerSeconds, setTimerSeconds] = useState(25 * 60);
  const [timerRunning, setTimerRunning] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);

  const today = localDateKey();
  const completeCount = tasks.filter((task) => task.completed).length;
  const openCount = tasks.length - completeCount;
  const completionPercent = tasks.length
    ? Math.round((completeCount / tasks.length) * 100)
    : 0;

  const navItems = [
    {
      key: 'today' as const,
      label: 'Today',
      icon: SunMedium,
      count: tasks.filter(
        (task) => !task.completed && (!task.dueDate || task.dueDate <= today),
      ).length,
    },
    {
      key: 'inbox' as const,
      label: 'Inbox',
      icon: Inbox,
      count: tasks.filter((task) => task.project === 'Inbox' && !task.completed).length,
    },
    {
      key: 'upcoming' as const,
      label: 'Upcoming',
      icon: CalendarDays,
      count: tasks.filter(
        (task) => !task.completed && Boolean(task.dueDate && task.dueDate > today),
      ).length,
    },
    { key: 'all' as const, label: 'All tasks', icon: ListTodo, count: tasks.length },
    {
      key: 'completed' as const,
      label: 'Completed',
      icon: CheckCircle2,
      count: completeCount,
    },
  ];

  const filteredTasks = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const priorityRank = { High: 0, Medium: 1, Low: 2 };

    return tasks
      .filter((task) => {
        if (view === 'today') {
          return !task.completed && (!task.dueDate || task.dueDate <= today);
        }
        if (view === 'inbox') return !task.completed && task.project === 'Inbox';
        if (view === 'upcoming') {
          return !task.completed && Boolean(task.dueDate && task.dueDate > today);
        }
        if (view === 'completed') return task.completed;
        if (view.startsWith('project:')) {
          return task.project === view.slice('project:'.length);
        }
        return true;
      })
      .filter((task) =>
        normalizedQuery
          ? [task.title, task.notes, task.project, task.priority]
              .join(' ')
              .toLowerCase()
              .includes(normalizedQuery)
          : true,
      )
      .sort((a, b) => {
        if (sort === 'title') return a.title.localeCompare(b.title);
        if (sort === 'due') return (a.dueDate ?? '9999').localeCompare(b.dueDate ?? '9999');
        return priorityRank[a.priority] - priorityRank[b.priority];
      });
  }, [query, sort, tasks, today, view]);

  const upcomingTasks = tasks
    .filter((task) => !task.completed && Boolean(task.dueDate && task.dueDate > today))
    .sort((a, b) => (a.dueDate ?? '').localeCompare(b.dueDate ?? ''))
    .slice(0, 2);
  const focusTask = tasks.find((task) => !task.completed) ?? null;

  const viewTitle = view.startsWith('project:')
    ? view.slice('project:'.length)
    : navItems.find((item) => item.key === view)?.label ?? 'Tasks';

  useEffect(() => {
    if (!timerRunning) return;
    const interval = window.setInterval(() => {
      setTimerSeconds((seconds) => {
        if (seconds <= 1) {
          setTimerRunning(false);
          return 0;
        }
        return seconds - 1;
      });
    }, 1000);
    return () => window.clearInterval(interval);
  }, [timerRunning]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === '/' && document.activeElement?.tagName !== 'INPUT') {
        event.preventDefault();
        searchRef.current?.focus();
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  async function addTask() {
    const title = draft.trim();
    if (!title || saving) return;
    setSaving(true);
    setMutationError(null);
    try {
      await createTask({
        title,
        project: 'Inbox',
        priority: 'Medium',
        dueDate: today,
        estimateMinutes: 25,
      });
      setDraft('');
      setView('today');
    } catch (cause) {
      setMutationError(cause instanceof Error ? cause.message : 'Unable to add task.');
    } finally {
      setSaving(false);
    }
  }

  async function toggleTask(task: Task) {
    setMutationError(null);
    try {
      await updateTask(task.id, { completed: !task.completed });
    } catch (cause) {
      setMutationError(cause instanceof Error ? cause.message : 'Unable to update task.');
    }
  }

  function openEditor(task: Task) {
    setEditing(task);
    setEditDraft(taskDraft(task));
  }

  async function saveEdit() {
    if (!editing || !editDraft?.title.trim()) return;
    setSaving(true);
    setMutationError(null);
    try {
      await updateTask(editing.id, {
        title: editDraft.title,
        notes: editDraft.notes,
        project: editDraft.project,
        priority: editDraft.priority,
        dueDate: editDraft.dueDate || null,
        estimateMinutes: Number(editDraft.estimateMinutes) || 25,
      });
      setEditing(null);
      setEditDraft(null);
    } catch (cause) {
      setMutationError(cause instanceof Error ? cause.message : 'Unable to save task.');
    } finally {
      setSaving(false);
    }
  }

  async function confirmDelete() {
    if (!deleting) return;
    setSaving(true);
    setMutationError(null);
    try {
      await removeTask(deleting.id);
      setDeleting(null);
    } catch (cause) {
      setMutationError(cause instanceof Error ? cause.message : 'Unable to delete task.');
    } finally {
      setSaving(false);
    }
  }

  return (
    <main className="min-h-screen overflow-x-hidden bg-background text-foreground">
      <div className="grid min-h-screen lg:grid-cols-[252px_minmax(0,1fr)]">
        <aside className="hidden border-r border-sidebar-border bg-sidebar px-4 py-5 text-sidebar-foreground lg:flex lg:flex-col">
          <div className="mb-8 flex items-center justify-between px-2">
            <div className="flex items-center gap-3">
              <div className="grid size-9 place-items-center rounded-xl bg-[#c9ff5f] text-[#10120f] shadow-[0_8px_24px_rgba(201,255,95,0.18)]">
                <Check className="size-5 stroke-[3]" />
              </div>
              <div>
                <p className="text-base font-semibold tracking-[-0.03em] text-white">
                  Zenith
                </p>
                <p className="text-xs text-white/45">Personal workspace</p>
              </div>
            </div>
            <Button
              aria-label="Workspace settings"
              className="text-white/45 hover:bg-white/8 hover:text-white"
              size="icon-sm"
              variant="ghost"
            >
              <Settings2 />
            </Button>
          </div>

          <nav aria-label="Task views" className="space-y-1">
            {navItems.map((item) => (
              <button
                key={item.key}
                className={cn(
                  'flex h-10 w-full items-center gap-3 rounded-xl px-3 text-left text-sm transition-colors',
                  view === item.key
                    ? 'bg-white/10 font-medium text-white'
                    : 'text-white/55 hover:bg-white/6 hover:text-white',
                )}
                onClick={() => setView(item.key)}
                type="button"
              >
                <item.icon className="size-[17px]" />
                <span className="flex-1">{item.label}</span>
                <span className="text-xs tabular-nums text-white/35">{item.count}</span>
              </button>
            ))}
          </nav>

          <div className="mt-8 flex items-center justify-between px-3">
            <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white/30">
              Projects
            </p>
            <Button
              aria-label="Add project"
              className="text-white/40 hover:bg-white/8 hover:text-white"
              size="icon-xs"
              variant="ghost"
            >
              <Plus />
            </Button>
          </div>
          <div className="mt-2 space-y-1">
            {projects.map((project) => (
              <button
                key={project.name}
                className={cn(
                  'flex h-9 w-full items-center gap-3 rounded-xl px-3 text-sm transition-colors hover:bg-white/6 hover:text-white',
                  view === `project:${project.name}` ? 'text-white' : 'text-white/55',
                )}
                onClick={() => setView(`project:${project.name}`)}
                type="button"
              >
                <span className={cn('size-2 rounded-full', project.color)} />
                <span className="flex-1 text-left">{project.name}</span>
                <span className="text-xs text-white/30">
                  {tasks.filter((task) => task.project === project.name).length}
                </span>
              </button>
            ))}
          </div>

          <div className="mt-auto rounded-2xl border border-white/8 bg-white/[0.045] p-4">
            <div className="mb-3 flex items-center gap-2 text-[#c9ff5f]">
              <Sparkles className="size-4" />
              <span className="text-xs font-semibold uppercase tracking-[0.1em]">
                Workspace pulse
              </span>
            </div>
            <p className="text-sm leading-5 text-white/65">
              {completeCount} complete · {openCount} ready to move forward.
            </p>
            <div className="mt-4 flex h-16 items-end gap-1.5" aria-label="Task momentum">
              {[38, 55, 44, 67, 50, 82, 62].map((height, index) => (
                <span
                  key={`${height}-${index}`}
                  className={cn(
                    'flex-1 rounded-sm',
                    index === 5 ? 'bg-[#c9ff5f]' : 'bg-white/14',
                  )}
                  style={{ height: `${height}%` }}
                />
              ))}
            </div>
          </div>
        </aside>

        <section className="min-w-0 max-w-full">
          <header className="sticky top-0 z-30 flex h-[72px] items-center gap-3 border-b bg-white/82 px-5 backdrop-blur-xl md:px-8 xl:px-10">
            <div className="flex items-center gap-3 lg:hidden">
              <div className="grid size-9 place-items-center rounded-xl bg-foreground text-background">
                <Check className="size-5 stroke-[3]" />
              </div>
              <span className="font-semibold">Zenith</span>
            </div>
            <div className="relative ml-auto hidden w-full max-w-[340px] md:block lg:ml-0">
              <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                ref={searchRef}
                aria-label="Search tasks"
                className="h-10 border-transparent bg-muted/70 pl-9 pr-12 shadow-none focus-visible:bg-white"
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search tasks, projects, notes…"
                value={query}
              />
              <kbd className="absolute right-3 top-1/2 hidden -translate-y-1/2 rounded border bg-white px-1.5 py-0.5 text-[10px] text-muted-foreground lg:block">
                /
              </kbd>
            </div>
            <div className="ml-auto flex items-center gap-2">
              <Button aria-label="Notifications" size="icon" variant="ghost">
                <Bell />
              </Button>
              <div className="ml-1 grid size-9 place-items-center rounded-full bg-[#3f37c9] text-sm font-semibold text-white ring-4 ring-[#3f37c9]/10">
                DS
              </div>
            </div>
          </header>

          <nav
            aria-label="Mobile task views"
            className="sticky top-[72px] z-20 flex gap-2 overflow-x-auto border-b bg-background/95 px-5 py-3 backdrop-blur lg:hidden"
          >
            {navItems.map((item) => (
              <Button
                key={item.key}
                className="rounded-full"
                onClick={() => setView(item.key)}
                size="sm"
                variant={view === item.key ? 'default' : 'outline'}
              >
                <item.icon />
                {item.label}
              </Button>
            ))}
          </nav>

          <div className="mx-auto grid w-full max-w-[1380px] gap-8 px-5 py-7 md:px-8 lg:py-9 xl:grid-cols-[minmax(0,1fr)_300px] xl:px-10">
            <section className="min-w-0">
              <div className="mb-7 flex flex-wrap items-end justify-between gap-4">
                <div>
                  <p className="mb-1 flex items-center gap-2 text-sm font-medium text-[#665cf6]">
                    <SunMedium className="size-4" />
                    {new Intl.DateTimeFormat('en-US', {
                      weekday: 'long',
                      month: 'long',
                      day: 'numeric',
                    }).format(new Date())}
                  </p>
                  <h1 className="text-3xl font-semibold tracking-[-0.045em] md:text-[2.55rem] md:leading-tight">
                    Make today count.
                  </h1>
                </div>
                <Button
                  className="h-10 rounded-xl px-4"
                  onClick={() => setLayout((current) => (current === 'list' ? 'board' : 'list'))}
                  variant="outline"
                >
                  {layout === 'list' ? <LayoutGrid /> : <List />}
                  {layout === 'list' ? 'Board view' : 'List view'}
                </Button>
              </div>

              <section className="relative mb-7 overflow-hidden rounded-[24px] bg-[#3f37c9] px-6 py-6 text-white shadow-[0_24px_60px_rgba(63,55,201,0.18)] md:px-7">
                <div className="absolute -right-16 -top-24 size-60 rounded-full border-[34px] border-white/[0.055]" />
                <div className="relative flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
                  <div>
                    <p className="text-sm font-medium text-white/65">Workspace progress</p>
                    <p className="mt-1 text-2xl font-semibold tracking-[-0.04em]">
                      {completeCount} of {tasks.length} tasks complete
                    </p>
                    <p className="mt-2 max-w-md text-sm leading-5 text-white/62">
                      {openCount > 0
                        ? `${openCount} task${openCount === 1 ? '' : 's'} remain. Choose one clear next move.`
                        : 'Your workspace is clear. Enjoy the breathing room.'}
                    </p>
                  </div>
                  <div className="flex items-center gap-4 md:min-w-[240px]">
                    <div className="flex-1">
                      <div className="mb-2 flex justify-between text-xs font-medium text-white/60">
                        <span>Momentum</span>
                        <span>{completionPercent}%</span>
                      </div>
                      <Progress
                        className="h-2.5 bg-white/15 [&_[data-slot=progress-indicator]]:bg-[#c9ff5f]"
                        value={completionPercent}
                      />
                    </div>
                    <div className="grid size-11 place-items-center rounded-2xl bg-white/10">
                      <Sparkles className="size-5 text-[#c9ff5f]" />
                    </div>
                  </div>
                </div>
              </section>

              <form
                className="mb-6 flex gap-2 rounded-2xl border bg-card p-2 shadow-[0_10px_34px_rgba(27,24,18,0.045)]"
                onSubmit={(event) => {
                  event.preventDefault();
                  void addTask();
                }}
              >
                <div className="grid size-10 shrink-0 place-items-center rounded-xl bg-[#c9ff5f]/45 text-[#29340d]">
                  <Plus className="size-5" />
                </div>
                <Input
                  aria-label="New task title"
                  className="h-10 min-w-0 flex-1 border-0 px-2 text-base shadow-none focus-visible:ring-0"
                  disabled={saving}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder="Add something you want to finish…"
                  value={draft}
                />
                <Button className="h-10 rounded-xl px-4" disabled={!draft.trim() || saving} type="submit">
                  {saving ? 'Saving…' : 'Add task'}
                </Button>
              </form>

              <div className="relative mb-5 md:hidden">
                <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  ref={searchRef}
                  aria-label="Search tasks"
                  className="h-10 bg-card pl-9"
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search this workspace…"
                  value={query}
                />
              </div>

              {error || mutationError ? (
                <Alert className="mb-5 border-[#f2b8ae] bg-[#fff5f3] text-[#8f2e22]" variant="destructive">
                  <AlertTitle>Something needs attention</AlertTitle>
                  <AlertDescription className="flex flex-wrap items-center justify-between gap-3">
                    <span>{error ?? mutationError}</span>
                    {error ? (
                      <Button onClick={() => void load()} size="sm" variant="outline">
                        Try again
                      </Button>
                    ) : null}
                  </AlertDescription>
                </Alert>
              ) : null}

              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-semibold tracking-[-0.025em]">{viewTitle}</h2>
                  <Badge className="bg-muted text-muted-foreground" variant="secondary">
                    {filteredTasks.length} {filteredTasks.length === 1 ? 'task' : 'tasks'}
                  </Badge>
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger
                    render={
                      <Button className="text-muted-foreground" size="sm" variant="ghost" />
                    }
                  >
                    Sort
                    <ChevronDown />
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-40">
                    <DropdownMenuItem onClick={() => setSort('smart')}>Priority</DropdownMenuItem>
                    <DropdownMenuItem onClick={() => setSort('due')}>Due date</DropdownMenuItem>
                    <DropdownMenuItem onClick={() => setSort('title')}>Task name</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              {loading ? (
                <div className="space-y-2.5" aria-label="Loading tasks">
                  {[0, 1, 2, 3].map((item) => (
                    <Skeleton key={item} className="h-[78px] rounded-2xl bg-card" />
                  ))}
                </div>
              ) : filteredTasks.length === 0 ? (
                <Empty className="min-h-56 border bg-card">
                  <EmptyHeader>
                    <EmptyMedia className="size-11 rounded-2xl bg-[#f0edff] text-[#5a4fd0]" variant="icon">
                      <CheckCircle2 />
                    </EmptyMedia>
                    <EmptyTitle>Nothing here right now</EmptyTitle>
                    <EmptyDescription>
                      {query
                        ? 'Try a different search phrase or clear the search.'
                        : 'Add a task above or choose another workspace view.'}
                    </EmptyDescription>
                  </EmptyHeader>
                </Empty>
              ) : (
                <div
                  className={cn(
                    'gap-2.5',
                    layout === 'list' ? 'space-y-2.5' : 'grid sm:grid-cols-2',
                  )}
                >
                  {filteredTasks.map((task) => (
                    <article
                      key={task.id}
                      className={cn(
                        'group flex gap-3 rounded-2xl border bg-card px-4 py-4 shadow-[0_8px_28px_rgba(27,24,18,0.035)] transition-all hover:-translate-y-0.5 hover:shadow-[0_14px_34px_rgba(27,24,18,0.07)]',
                        layout === 'list' ? 'items-start md:items-center' : 'min-h-36 items-start',
                        task.completed && 'bg-muted/35 opacity-70 shadow-none',
                      )}
                    >
                      <Checkbox
                        aria-label={`Mark ${task.title} as ${task.completed ? 'incomplete' : 'complete'}`}
                        checked={task.completed}
                        className="mt-0.5 size-5 rounded-full data-checked:border-[#3f37c9] data-checked:bg-[#3f37c9]"
                        onCheckedChange={() => void toggleTask(task)}
                      />
                      <div className="min-w-0 flex-1">
                        <p
                          className={cn(
                            'font-medium tracking-[-0.012em]',
                            task.completed && 'line-through',
                          )}
                        >
                          {task.title}
                        </p>
                        {layout === 'board' && task.notes ? (
                          <p className="mt-2 line-clamp-2 text-xs leading-5 text-muted-foreground">
                            {task.notes}
                          </p>
                        ) : null}
                        <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1.5">
                            <span
                              className={cn(
                                'size-2 rounded-full',
                                projectColors[task.project] ?? 'bg-zinc-400',
                              )}
                            />
                            {task.project}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock3 className="size-3.5" />
                            {task.estimateMinutes} min
                          </span>
                        </div>
                      </div>
                      <div
                        className={cn(
                          'flex shrink-0 items-center gap-2',
                          layout === 'board' && 'flex-col-reverse items-end self-stretch justify-between',
                        )}
                      >
                        <span
                          className={cn(
                            'hidden rounded-lg px-2 py-1 text-xs font-medium sm:inline-flex',
                            task.priority === 'High' && 'bg-[#ffebe7] text-[#c94532]',
                            task.priority === 'Medium' && 'bg-[#eeeafd] text-[#5a4fd0]',
                            task.priority === 'Low' && 'bg-[#e6f8f1] text-[#237b64]',
                          )}
                        >
                          {task.priority}
                        </span>
                        <span
                          className={cn(
                            'hidden min-w-[72px] text-right text-xs font-medium md:block',
                            task.dueDate && task.dueDate < today
                              ? 'text-[#c94532]'
                              : 'text-muted-foreground',
                          )}
                        >
                          {readableDate(task.dueDate)}
                        </span>
                        <DropdownMenu>
                          <DropdownMenuTrigger
                            render={
                              <Button
                                aria-label={`More options for ${task.title}`}
                                className="text-muted-foreground opacity-60 group-hover:opacity-100"
                                size="icon-sm"
                                variant="ghost"
                              />
                            }
                          >
                            <MoreHorizontal />
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="w-36">
                            <DropdownMenuItem onClick={() => openEditor(task)}>
                              <Pencil />
                              Edit
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem onClick={() => setDeleting(task)} variant="destructive">
                              <Trash2 />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </section>

            <aside className="hidden space-y-5 xl:block">
              <section className="rounded-[22px] border bg-card p-5 shadow-[0_12px_36px_rgba(27,24,18,0.04)]">
                <div className="mb-5 flex items-start justify-between">
                  <div>
                    <p className="text-sm font-semibold">Focus session</p>
                    <p className="mt-0.5 text-xs text-muted-foreground">
                      Quiet the noise for one task.
                    </p>
                  </div>
                  <div className="grid size-9 place-items-center rounded-xl bg-[#f0edff] text-[#5a4fd0]">
                    <TimerReset className="size-4" />
                  </div>
                </div>
                <div className="grid place-items-center rounded-2xl bg-[#f5f3ff] px-4 py-6 text-center">
                  <p className="font-mono text-[2.6rem] font-semibold tracking-[-0.06em] text-[#312b88]">
                    {formatTimer(timerSeconds)}
                  </p>
                  <p className="mt-1 line-clamp-1 text-xs font-medium uppercase tracking-[0.1em] text-[#655bb2]">
                    {timerSeconds === 0
                      ? 'Session complete'
                      : focusTask?.title ?? 'Ready when you are'}
                  </p>
                </div>
                <div className="mt-3 grid grid-cols-[1fr_auto] gap-2">
                  <Button
                    className="h-10 rounded-xl bg-[#3f37c9] hover:bg-[#312aaf]"
                    disabled={timerSeconds === 0}
                    onClick={() => setTimerRunning((running) => !running)}
                  >
                    {timerRunning ? <Pause /> : <Play />}
                    {timerRunning ? 'Pause' : 'Start focus'}
                  </Button>
                  <Button
                    aria-label="Reset focus timer"
                    className="h-10 w-10 rounded-xl"
                    onClick={() => {
                      setTimerRunning(false);
                      setTimerSeconds(25 * 60);
                    }}
                    variant="outline"
                  >
                    <RotateCcw />
                  </Button>
                </div>
              </section>

              <section className="rounded-[22px] border bg-card p-5 shadow-[0_12px_36px_rgba(27,24,18,0.04)]">
                <div className="mb-4 flex items-center justify-between">
                  <h2 className="text-sm font-semibold">Coming up</h2>
                  <Button onClick={() => setView('upcoming')} size="xs" variant="ghost">
                    View all
                  </Button>
                </div>
                {upcomingTasks.length ? (
                  <div className="space-y-4">
                    {upcomingTasks.map((task, index) => (
                      <div key={task.id} className="flex gap-3">
                        <div
                          className={cn(
                            'grid size-10 shrink-0 place-items-center rounded-xl',
                            index === 0
                              ? 'bg-[#fff2df] text-[#a86416]'
                              : 'bg-[#e7f8f2] text-[#237b64]',
                          )}
                        >
                          {index === 0 ? (
                            <CalendarDays className="size-4" />
                          ) : (
                            <Circle className="size-4 fill-current" />
                          )}
                        </div>
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium">{task.title}</p>
                          <p className="mt-0.5 text-xs text-muted-foreground">
                            {readableDate(task.dueDate)} · {task.estimateMinutes} min
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm leading-6 text-muted-foreground">
                    No upcoming deadlines. Your schedule has room.
                  </p>
                )}
              </section>
            </aside>
          </div>
        </section>
      </div>

      <Dialog
        open={Boolean(editing)}
        onOpenChange={(open) => {
          if (!open) {
            setEditing(null);
            setEditDraft(null);
          }
        }}
      >
        <DialogContent className="max-w-lg rounded-2xl p-5">
          <DialogHeader>
            <DialogTitle className="text-lg">Edit task</DialogTitle>
            <DialogDescription>
              Keep the next action specific enough to complete in one sitting.
            </DialogDescription>
          </DialogHeader>
          {editDraft ? (
            <div className="grid gap-4 py-2">
              <label
                className="grid gap-1.5 text-sm font-medium"
                htmlFor="edit-task-title"
              >
                Task title
                <Input
                  className="h-10"
                  id="edit-task-title"
                  onChange={(event) =>
                    setEditDraft((current) =>
                      current ? { ...current, title: event.target.value } : current,
                    )
                  }
                  value={editDraft.title}
                />
              </label>
              <label
                className="grid gap-1.5 text-sm font-medium"
                htmlFor="edit-task-notes"
              >
                Notes
                <Textarea
                  className="min-h-24 resize-none"
                  id="edit-task-notes"
                  onChange={(event) =>
                    setEditDraft((current) =>
                      current ? { ...current, notes: event.target.value } : current,
                    )
                  }
                  placeholder="Add context, links, or a clear definition of done."
                  value={editDraft.notes}
                />
              </label>
              <div className="grid gap-4 sm:grid-cols-2">
                <label
                  className="grid gap-1.5 text-sm font-medium"
                  htmlFor="edit-task-project"
                >
                  Project
                  <Select
                    onValueChange={(value) =>
                      setEditDraft((current) =>
                        current ? { ...current, project: value ?? 'Inbox' } : current,
                      )
                    }
                    value={editDraft.project}
                  >
                    <SelectTrigger className="h-10 w-full" id="edit-task-project">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Inbox">Inbox</SelectItem>
                      {projects.map((project) => (
                        <SelectItem key={project.name} value={project.name}>
                          {project.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </label>
                <label
                  className="grid gap-1.5 text-sm font-medium"
                  htmlFor="edit-task-priority"
                >
                  Priority
                  <Select
                    onValueChange={(value) =>
                      setEditDraft((current) =>
                        current
                          ? { ...current, priority: (value ?? 'Medium') as TaskPriority }
                          : current,
                      )
                    }
                    value={editDraft.priority}
                  >
                    <SelectTrigger className="h-10 w-full" id="edit-task-priority">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="High">High</SelectItem>
                      <SelectItem value="Medium">Medium</SelectItem>
                      <SelectItem value="Low">Low</SelectItem>
                    </SelectContent>
                  </Select>
                </label>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                <label
                  className="grid gap-1.5 text-sm font-medium"
                  htmlFor="edit-task-due-date"
                >
                  Due date
                  <Input
                    className="h-10"
                    id="edit-task-due-date"
                    onChange={(event) =>
                      setEditDraft((current) =>
                        current ? { ...current, dueDate: event.target.value } : current,
                      )
                    }
                    type="date"
                    value={editDraft.dueDate}
                  />
                </label>
                <label
                  className="grid gap-1.5 text-sm font-medium"
                  htmlFor="edit-task-estimate"
                >
                  Estimate (minutes)
                  <Input
                    className="h-10"
                    id="edit-task-estimate"
                    max={480}
                    min={5}
                    onChange={(event) =>
                      setEditDraft((current) =>
                        current
                          ? { ...current, estimateMinutes: event.target.value }
                          : current,
                      )
                    }
                    type="number"
                    value={editDraft.estimateMinutes}
                  />
                </label>
              </div>
            </div>
          ) : null}
          <DialogFooter className="-mx-5 -mb-5 px-5">
            <Button
              onClick={() => {
                setEditing(null);
                setEditDraft(null);
              }}
              variant="outline"
            >
              Cancel
            </Button>
            <Button disabled={!editDraft?.title.trim() || saving} onClick={() => void saveEdit()}>
              {saving ? 'Saving…' : 'Save changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={Boolean(deleting)} onOpenChange={(open) => !open && setDeleting(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this task?</AlertDialogTitle>
            <AlertDialogDescription>
              “{deleting?.title}” will be permanently removed from the workspace.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep task</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-white hover:bg-destructive/90"
              disabled={saving}
              onClick={() => void confirmDelete()}
            >
              Delete task
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </main>
  );
}
