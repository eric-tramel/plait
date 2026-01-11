---
description: Find and start working on the highest priority open issue
allowed-tools: Task, Read, Glob, Grep, Bash(git:*), TodoWrite
---

# Start High Priority Issue

Find and start working on the highest priority open issue from the issue tracker.

## Instructions

### Step 1: Find Highest Priority Issue

Use the top-issue-agent (Task tool with subagent_type='top-issue-agent') to find the highest priority open issue:

Prompt the agent: "Find the highest priority open issue by analyzing the issues/ directory. Look at all open issues, analyze their dependencies (blocked_by fields), and return the filepath of the issue that should be worked on next - specifically one that has no blockers and blocks the most other issues."

The agent will return a priority analysis and the filepath of the selected issue. Priority is determined by:

1. **No unresolved dependencies** - The issue cannot have any "Depends on" issues that are still open
2. **Blocks the most** - Among issues with no dependencies, prefer those that block other issues
3. **Lower issue number** - If still tied, prefer lower issue numbers (created earlier)

### Step 2: Check Working Directory State

Before creating a branch, check if the working directory is clean:

```bash
git status --porcelain
```

If there are uncommitted changes, warn the user and ask if they want to proceed or stash changes first.

### Step 3: Create Feature Branch

Follow the workflow from TASKS.md:

```bash
# Ensure we're on main and up to date
git checkout main
git pull

# Create feature branch with descriptive name
git checkout -b feat/<issue-slug>
```

Branch naming convention:
- Use the issue slug from the filename (e.g., `1-empty-parameter-updates.md` → `feat/empty-parameter-updates`)
- Remove the issue number prefix from the slug

### Step 4: Read Design Docs and Context

Before starting implementation:
1. Read any design docs mentioned in the issue's Notes section
2. Read related code files mentioned in the issue
3. Understand acceptance criteria thoroughly

### Step 5: Create Implementation Plan

Create a todo list with implementation steps based on acceptance criteria:
- Break down each acceptance criterion into actionable tasks
- Include test writing tasks
- Include CHANGELOG update task
- Include `make ci` verification task

Use the TodoWrite tool to track these tasks.

### Step 6: Begin Implementation

Start working on the first task in the plan. For each task:
1. Mark it as in_progress in the todo list
2. Implement the change
3. Mark as completed when done
4. Move to next task

### Step 7: Open New Issues for Discovered Work

During implementation, you may discover:
- Bugs in existing code
- Missing functionality that's out of scope for the current issue
- Technical debt or refactoring opportunities
- Follow-up enhancements

For each discovered item, use `/new-issue` to create a tracking issue. This ensures work is not lost and can be prioritized appropriately.

## Output Format

Report progress as you go:

```
## Starting Issue #2: Align Optimizer with Value.ref System

Branch created: feat/align-optimizer-with-value-ref

### Implementation Plan
1. [ ] Review design_docs/optimization.md
2. [ ] Review design_docs/values.md
3. [ ] Update VerifierLoss for Value objects
4. [ ] Update SFAOptimizer for ValueRef handling
5. [ ] Update backward pass ValueRef resolution
6. [ ] Add integration tests
7. [ ] Update CHANGELOG.md
8. [ ] Run make ci
9. [ ] Open new issues for any discovered work (use /new-issue)

### Starting Task 1: Review design_docs/optimization.md
...
```

## Edge Cases

- **Dirty working directory**: Warn user, ask to stash or commit before proceeding
- **All issues have dependencies**: Report circular or blocked dependency situation
- **No open issues**: Report "No open issues found. Nothing to work on!"
- **Branch already exists**: Ask user if they want to switch to it or create a new name
