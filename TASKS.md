# TASKS.md

A line-by-line breakdown of PRs to implement inf-engine, in order.

Each PR represents a single, tested, reviewable increment of functionality.

## Progress

- [ ] **Phase 5: Values + Functional + Tracing/Execution** (0/5)

**Total: 0/5 PRs completed**

---

## PR Requirements

Every PR must:
- [ ] Create a feature branch from `main`
- [ ] Include implementation code
- [ ] Include unit tests (100% coverage of new code)
- [ ] Include integration tests where applicable
- [ ] Update `CHANGELOG.md`
- [ ] Pass `make ci`
- [ ] Include usage examples in docstrings or tests

---

## Phase 5: Values + Functional + Tracing/Execution

### - [ ] PR-059: Values core (Value, ValueKind, helpers, ValueRef)
- **Branch**: `feat/values-core`
- **Description**: Implement `Value`, `ValueKind`, and core helpers (`valueify`, `unwrap`, `collect_refs`, `ValueRef`, `replace_values_with_refs`). Establish stable parameter refs (`param:module.path.name`) and input refs (`input:<name>`).
- **Design Docs**:
  - `values.md` → "Core Data Model", "Construction Helpers", "ValueRef"
  - `parameters.md` → "Parameter vs Value", "Identity and Naming"
- **Files**:
  - `src/inf_engine/values.py` (Value, ValueKind, valueify/unwrap/collect_refs, ValueRef)
- **Tests**:
  - `tests/unit/test_values.py` (valueify/unwrap, collect_refs on nested structures, ValueRef replacement)
  - `tests/unit/test_valueify_parameters.py` (Parameter lifted to Value with stable ref, metadata)
- **CHANGELOG**: "Add Value container and ValueRef helpers"

### - [ ] PR-060: Parameter lifting + refs
- **Branch**: `feat/parameter-lifting`
- **Description**: Update Parameter usage to lift into `Value` with stable refs and metadata; ensure `Parameter` description requirements are enforced with structured values.
- **Design Docs**:
  - `parameters.md` → "Lifting Parameters into Values", "Identity and Naming"
  - `values.md` → "Parameters vs Values"
- **Files**:
  - `src/inf_engine/parameter.py` (metadata hooks, helpers if needed)
  - `src/inf_engine/values.py` (valueify(Parameter) behavior)
- **Tests**:
  - `tests/unit/test_parameter.py` (description required when requires_grad=True)
  - `tests/unit/test_parameter_value_refs.py` (ref format, structured param kind inference)
- **CHANGELOG**: "Lift parameters into Value with stable refs"

### - [ ] PR-061: Functional API (stateless ops)
- **Branch**: `feat/functional-api`
- **Description**: Implement `inf_engine.functional` ops with Value-based error semantics (no exceptions), including `render`, `select`, `parse_structured`, `concat`, and error propagation rules.
- **Design Docs**:
  - `functional_api.md` → "Function Reference", "Error Propagation and Resolution"
  - `values.md` → "Structured Access (getitem)"
- **Files**:
  - `src/inf_engine/functional.py` (functional ops)
- **Tests**:
  - `tests/unit/test_functional_text.py` (render/concat/format, Value(ERROR) pass-through)
  - `tests/unit/test_functional_structured.py` (select path, structured chaining, parse_structured errors)
  - `tests/unit/test_functional_errors.py` (error resolution precedence, default handling)
- **CHANGELOG**: "Add functional API for Value operations"

### - [ ] PR-062: Tracing with Values + ValueRef
- **Branch**: `feat/value-tracing`
- **Description**: Update tracing to Value-driven capture: bind inputs, collect dependencies via `Value.ref`, store ValueRef placeholders in args/kwargs, and output IDs via `collect_refs`.
- **Design Docs**:
  - `tracing.md` → "Value-Driven Capture", "ValueRef"
  - `values.md` → "ValueRef"
- **Files**:
  - `src/inf_engine/tracing/tracer.py` (bind_inputs, record_call, replace_values_with_refs)
  - `src/inf_engine/tracing/context.py` (no change, for completeness)
- **Tests**:
  - `tests/unit/test_tracer_values.py` (dependencies from Value.ref, nested structures)
  - `tests/integration/test_tracing_values.py` (simple module graph, output_ids from Values)
- **CHANGELOG**: "Refactor tracing to Value-driven capture with ValueRef"

### - [ ] PR-063: Execution with ValueRef + error Values
- **Branch**: `feat/value-execution`
- **Description**: Update execution to resolve ValueRef placeholders, unwrap Value payloads before forward(), and propagate Value(ERROR) per functional API rules.
- **Design Docs**:
  - `execution.md` → "ValueRef placeholders", "Value(ERROR) outputs"
  - `values.md` → "ValueRef"
- **Files**:
  - `src/inf_engine/execution/state.py` (resolve ValueRef in args/kwargs)
  - `src/inf_engine/execution/scheduler.py` (Value(ERROR) short-circuit behavior)
  - `src/inf_engine/execution/executor.py` (unwrap Values for forward())
- **Tests**:
  - `tests/unit/test_execution_value_ref.py` (resolve ValueRef)
  - `tests/unit/test_execution_errors.py` (Value(ERROR) short-circuit)
  - `tests/integration/test_execution_values.py` (end-to-end graph with structured select + error)
- **CHANGELOG**: "Execute graphs with ValueRef and error-as-value semantics"

---

## Summary

| Phase | PRs | Range |
|-------|-----|-------|
| Values + Functional + Tracing/Execution | 5 | PR-059 to PR-063 |
| **Total** | **5** | |

---

## Workflow

For each PR:

```bash
# 1. Create branch
git checkout main
git pull
git checkout -b feat/feature-name

# 2. Read referenced design docs
# Review the Design Docs sections listed in the PR

# 3. Implement
# ... write code ...

# 4. Test
make ci

# 5. Update CHANGELOG
# Add entry under [Unreleased]

# 6. Commit
git add .
git commit -m "feat: description"

# 7. Push and create PR
git push -u origin feat/feature-name
gh pr create --title "PR-XXX: Description" --body "..."

# 8. After review and merge
git checkout main
git pull
```
