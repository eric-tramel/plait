---
description: Implement a GitHub issue using the developer-main agent
allowed-tools: Task
argument-hint: <issue number>
---

Use the Task tool to invoke the `developer-main` agent to implement GitHub issue #$ARGUMENTS.

The agent will handle the full workflow: gathering context, planning, creating a feature branch, implementing, testing, reviewing, and creating a PR.
