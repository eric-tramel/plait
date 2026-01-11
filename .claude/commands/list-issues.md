---
description: List all issues related to a topic
allowed-tools: Read, Glob, Grep
argument-hint: <topic or keyword>
model: haiku
---

## Task

Search and list all issues (open and closed) related to the user's topic.

## Context Gathering

Use the Glob tool to find all issues:
- Pattern: `issues/*.md` for open issues
- Pattern: `issues/closed/*.md` for closed issues

## Instructions

1. **Search for matching issues**: Look through all issue files in both `issues/` and `issues/closed/` for the topic `$ARGUMENTS`. Search should check:
   - Filename (the slug often contains keywords)
   - Issue title
   - Description content
   - Acceptance criteria
   - Notes section

2. **Read and analyze matches**: For each potential match, read the issue to confirm relevance and extract key details.

3. **Present results in a table format**:

   ```
   | #   | Title                      | Status | Related To |
   |-----|----------------------------|--------|------------|
   | 3   | Add async batch processing | open   | #1, #5     |
   | 7   | Fix batch timeout bug      | closed | #3         |
   ```

   Include:
   - Issue number
   - Title
   - Status (open/closed)
   - Related issues (from the "Related Issues" section if present)

4. **Handle no matches**: If no issues match the topic, tell the user and suggest:
   - Alternative search terms based on existing issue titles
   - Creating a new issue with `/new-issue`

5. **Summarize**: Provide a brief count (e.g., "Found 3 issues related to 'batch processing': 2 open, 1 closed")

## Topic to Search

$ARGUMENTS
