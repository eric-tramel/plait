---
description: Close an issue by moving it to issues/closed/
allowed-tools: Bash(mkdir:*), Bash(mv:*), Read, Edit, Glob
argument-hint: <#issue-number or issue filename>
---

## Task

Close an issue by moving it from `issues/` to `issues/closed/`.

## Context Gathering

Use the Glob tool to find existing issues:
- Pattern: `issues/*.md` for open issues
- Pattern: `issues/closed/*.md` for closed issues

## Instructions

1. **Identify the issue**: Match the user's input (`$ARGUMENTS`) to an existing issue file in `issues/`. The user may provide:
   - Just a number (e.g., `3` or `#3`)
   - A partial filename (e.g., `add-dark-mode`)
   - The full filename (e.g., `3-add-dark-mode.md`)

2. **Verify the issue exists**: If no matching issue is found, list available open issues and ask for clarification.

3. **Create the closed directory** if it doesn't exist:
   ```bash
   mkdir -p issues/closed
   ```

4. **Update the issue status**: Before moving, edit the issue file to change:
   - `**Status:** open` → `**Status:** closed`
   - Add a `**Closed:** <today's date in YYYY-MM-DD format>` line after the Status line

5. **Move the issue file**:
   ```bash
   mv issues/<filename>.md issues/closed/
   ```

6. **Confirm closure**: Tell the user the issue was closed and moved, including the issue number and title.

## User Input

$ARGUMENTS
