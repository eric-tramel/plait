# Design Revisions / TODOs

Notes capturing agreed simplifications and outstanding design debt.

## Simplifications (Branching Deferred)

- [x] Remove branching concepts and examples from design docs.
- [x] Remove branching phase from the development plan.
- [x] Purge branching PRs from `TASKS.md` and update totals.

## Consistency and API Alignment

- [x] Harmonize `Parameter` definition across docs: `description` is required when
  `requires_grad=True`, optional otherwise.

## Core Design TODOs (from review)

- [ ] Dependency encoding: replace string node IDs in args/kwargs with
  `ValueRef` placeholders and pytree traversal; support nested containers and
  avoid collisions with literal strings.
- [ ] Execution queue correctness: skip stale or cancelled tasks on dequeue
  (or add per-node generations) so invalidated tasks can’t run.
- [ ] Inputs in graphs: decouple concrete input values from traced graphs to
  enable graph reuse and avoid persisting sensitive data in checkpoints.
- [ ] ExecutionManager semantics: either return a `Future` when queued or
  rename `submit()` to reflect blocking behavior.
- [ ] Structured access: ensure `Value.__getitem__` and `F.select` are fully
  specified and traced, with deterministic error propagation.
- [ ] Torch parity vs async reality: document the sync/async boundary clearly
  to set expectations for “1-1 parity.”

## Deferred (If Branching Is Reintroduced)

- [ ] Runtime branch gating: ensure only the selected branch executes; do not
  enqueue both branches when a condition is evaluated at runtime.
