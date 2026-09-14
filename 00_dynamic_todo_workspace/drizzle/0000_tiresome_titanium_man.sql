CREATE TABLE `tasks` (
	`id` text PRIMARY KEY NOT NULL,
	`title` text NOT NULL,
	`notes` text DEFAULT '' NOT NULL,
	`project` text DEFAULT 'Inbox' NOT NULL,
	`priority` text DEFAULT 'Medium' NOT NULL,
	`due_date` text,
	`estimate_minutes` integer DEFAULT 25 NOT NULL,
	`completed` integer DEFAULT false NOT NULL,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `idx_tasks_completed_due` ON `tasks` (`completed`,`due_date`);--> statement-breakpoint
CREATE INDEX `idx_tasks_project` ON `tasks` (`project`);