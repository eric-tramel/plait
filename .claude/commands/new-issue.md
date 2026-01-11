---
description: Create a new issue in issues/ directory (GitHub-style local tracking)
allowed-tools: Bash(mkdir:*), Read, Write, Glob
argument-hint: <issue title or description>
---

## Task

Create a new issue file in the `issues/` directory based on the user's request.

## Context Gathering

Before creating the issue, use the Glob tool to find existing issues:
- Pattern: `issues/*.md` for open issues
- Pattern: `issues/closed/*.md` for closed issues

Then use the Read tool to examine any relevant existing issues for context and to determine the next issue number.

## Instructions

1. **Determine the next issue number**: Look at existing issue files in `issues/` and find the highest number. The new issue should be the next sequential number. If no issues exist, start at 1.

2. **Generate a slug from the title**: Create a kebab-case slug from the main topic (e.g., "Add dark mode support" becomes "add-dark-mode-support"). Keep it concise (3-5 words max).

3. **Create the issue file**: The filename format is `issues/<number>-<slug>.md`

4. **Identify related issues**: Review existing issues from the context above. Look for:
   - **Depends on**: Issues that must be completed before this one can start
   - **Blocks**: Issues that are waiting on this one
   - **Related to**: Issues that share context, components, or goals (but no dependency)

   Use the format `#<number>` to reference issues (e.g., `#3`, `#12`).

5. **Use this template for the issue content** (only include "Related Issues" section if there are actual links):

```markdown
# <Issue Title>

**Issue:** #<number>
**Created:** <today's date in YYYY-MM-DD format>
**Status:** open

## Related Issues

- **Depends on:** <#issue numbers this depends on, or "None">
- **Blocks:** <#issue numbers waiting on this, or "None">
- **Related to:** <#issue numbers with shared context, or "None">

## Description

<Expand on the user's request into a clear problem statement or feature description>

## Acceptance Criteria

- [ ] <Specific, testable criterion 1>
- [ ] <Specific, testable criterion 2>
- [ ] <Add more as needed>

## Notes

<Any additional context, constraints, or implementation hints>
```

6. **Confirm creation**: After creating the file, confirm the issue number, filename, and any linked issues to the user.

## User Request

$ARGUMENTS
