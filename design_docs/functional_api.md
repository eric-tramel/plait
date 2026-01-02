# Functional API

This document defines the functional, graph-aware API (similar to
`torch.nn.functional`) for operating on `Value` containers without defining
custom `InferenceModule` subclasses.

## Goals

1. **Stateless transforms**: Provide simple, composable operations.
2. **Graph-aware by default**: Calls record nodes during tracing.
3. **Value-first**: Inputs can be raw data or `Value`; outputs are `Value`.
4. **Explicit collections**: Core ops are scalar. Use `map_values`/`zip_values`/
   `stack_structured` for element-wise or batch workflows.

## Usage

```python
import inf_engine.functional as F
from inf_engine.values import Value, ValueKind, valueify

template = Value(ValueKind.FSTRING, "Summarize: {text}")
vars = Value(ValueKind.STRUCTURED, {"text": "long doc"})

prompt = F.render(template, vars)
summary = llm(prompt)
```

## Tracing and Execution Semantics

- **Tracing**: Each functional op records a graph node and returns `Value`
  with `ref` pointing to that node.
- **Execution**: Ops operate on `payload` (after `unwrap()`), then return a new
  `Value` with the appropriate `kind` and optional metadata.
- **Errors as values**: Ops never raise runtime errors. Failures return
  `Value(ERROR)` for graph-aware error propagation.
- **Collections**: Scalar ops do not traverse lists/tuples/dicts. Use
  `map_values`/`zip_values` for element-wise logic; `stack_structured` handles
  structured batching. Structured ops (`select`, `merge`) operate on structured
  payloads only.

## Error Propagation and Resolution

### Propagation Rules

1. **Short-circuit**: If any input is `Value(ERROR)`, the op returns an error
   without executing, unless the op is explicitly an error handler
   (e.g., `unwrap_or`, `is_error`).
2. **Deterministic resolution**: If multiple errors are present, the op returns
   a single “primary” error with the rest attached as causes.
3. **Op failures**: If the op itself fails (e.g., parse error), return a new
   `Value(ERROR)` with `meta` describing the op and input refs.

### Source Error Resolution

Errors are resolved in a deterministic traversal order:

- Positional args left-to-right
- Then kwargs sorted by key
- Lists/tuples left-to-right
- Dicts by sorted key order

The first error encountered is the **primary** error. Additional errors are
attached to the primary error under:

```python
error.meta["causes"] = [{"ref": "...", "message": "..."}]
```

This preserves provenance while keeping a single error value flowing.

## Function Reference

Unless stated otherwise, functions accept raw Python values or `Value` and
operate on a single value at a time. They do not implicitly map over
collections.

### Text and F-String Ops

#### F.render

**Signature**: `render(template: Value[FSTRING] | str, vars: Value[STRUCTURED] | dict[str, Any]) -> Value[TEXT] | Value[ERROR]`

**Description**: Render an f-string template using structured variables.

**Usage**:
```python
tmpl = valueify("Summarize: {text}", kind=ValueKind.FSTRING)
vars = valueify({"text": "doc"})
prompt = F.render(tmpl, vars)
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if template formatting fails (missing keys, invalid format).

#### F.concat

**Signature**: `concat(*parts: Value | str, sep: str = "") -> Value[TEXT] | Value[ERROR]`

**Description**: Join text-like inputs with an optional separator.

**Usage**:
```python
title = valueify("Title: ")
body = valueify("Hello")
text = F.concat(title, body)
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if any part cannot be coerced to text.

#### F.format

**Signature**: `format(fmt: Value[TEXT] | str, vars: Value[STRUCTURED] | dict[str, Any]) -> Value[TEXT] | Value[ERROR]`

**Description**: Apply structured formatting to a format string.

**Usage**:
```python
fmt = valueify("Name: {name}")
vars = valueify({"name": "Ada"})
text = F.format(fmt, vars)
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` on format failure.

#### F.strip

**Signature**: `strip(text: Value[TEXT] | str, *, chars: Value[TEXT] | str | None = None) -> Value[TEXT] | Value[ERROR]`

**Description**: Trim whitespace or specified characters from both ends.

**Usage**:
```python
clean = F.strip(valueify("  hi  "))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if input is not text-like.

#### F.lower / F.upper

**Signature**: `lower(text: Value[TEXT] | str) -> Value[TEXT] | Value[ERROR]`  
**Signature**: `upper(text: Value[TEXT] | str) -> Value[TEXT] | Value[ERROR]`

**Description**: Convert text to lowercase or uppercase.

**Usage**:
```python
lowered = F.lower(valueify("Hi"))
uppered = F.upper(valueify("Hi"))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if input is not text-like.

### Structured Ops

#### F.parse_structured

**Signature**: `parse_structured(text: Value[TEXT] | str, schema: type | None = None) -> Value[STRUCTURED] | Value[ERROR]`

**Description**: Parse structured text into a structured payload, optionally validating a schema.

**Usage**:
```python
structured = F.parse_structured(valueify("{\"a\": 1}"))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` on parse failure or schema validation failure.

#### F.select

**Signature**: `select(struct: Value[STRUCTURED] | dict | list | tuple, key_or_path: str | int | list[str | int], *, default: Any | None = None) -> Value | Value[ERROR]`

**Description**: Select a field by key or path from a structured value.

**Usage**:
```python
entity = F.select(valueify({"a": {"b": 1}}), "a.b")
name = F.select(valueify({"user": {"name": "Ada"}}), ["user", "name"])
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if the path is missing or invalid (unless `default` is provided).

**Chaining**:
If the selected value is structured, `F.select` returns `Value(STRUCTURED)` so
chained access remains graph-aware (e.g., `value["a"]["b"]`).

#### F.merge

**Signature**: `merge(*structs: Value[STRUCTURED] | dict[str, Any]) -> Value[STRUCTURED] | Value[ERROR]`

**Description**: Merge structured objects left-to-right.

**Usage**:
```python
merged = F.merge(valueify({"a": 1}), valueify({"b": 2}))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if any input is not structured.

#### F.coerce

**Signature**: `coerce(value: Value | str | int | float | dict | list | tuple, kind: ValueKind) -> Value | Value[ERROR]`

**Description**: Coerce a value to a target kind when safe.

**Coercion rules (required)**:
- `TEXT` → `INT` / `FLOAT`: parse numeric text.
- `INT` → `FLOAT`: widen.
- `FLOAT` → `INT`: only if `float.is_integer()` or explicit `meta["allow_lossy"]=True`.
- `STRUCTURED` → `TEXT`: serialize via stable JSON; reject if non-serializable.
- `TEXT` → `STRUCTURED`: only via `parse_structured` (use that instead).
- `ERROR` → any: return the same error.
- Unknown or unsafe conversions return `Value(ERROR)`.

**Usage**:
```python
text = F.coerce(valueify({"a": 1}), ValueKind.TEXT)
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if coercion is unsafe or unsupported.

### Numeric Ops

#### F.add

**Signature**: `add(a: Value[INT|FLOAT] | int | float, b: Value[INT|FLOAT] | int | float) -> Value[INT|FLOAT] | Value[ERROR]`

**Description**: Add two numeric values with numeric promotion.

**Usage**:
```python
sum_val = F.add(valueify(1), valueify(2.5))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if inputs are non-numeric.

#### F.sub

**Signature**: `sub(a: Value[INT|FLOAT] | int | float, b: Value[INT|FLOAT] | int | float) -> Value[INT|FLOAT] | Value[ERROR]`

**Description**: Subtract two numeric values with numeric promotion.

**Usage**:
```python
diff = F.sub(valueify(10), valueify(3))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if inputs are non-numeric.

#### F.mul

**Signature**: `mul(a: Value[INT|FLOAT] | int | float, b: Value[INT|FLOAT] | int | float) -> Value[INT|FLOAT] | Value[ERROR]`

**Description**: Multiply two numeric values with numeric promotion.

**Usage**:
```python
prod = F.mul(valueify(3), valueify(4))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if inputs are non-numeric.

#### F.div

**Signature**: `div(a: Value[INT|FLOAT] | int | float, b: Value[INT|FLOAT] | int | float) -> Value[FLOAT] | Value[ERROR]`

**Description**: Divide two numeric values (always returns FLOAT).

**Usage**:
```python
quot = F.div(valueify(7), valueify(2))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if inputs are non-numeric or divisor is zero.

#### F.sum

**Signature**: `sum(values: Sequence[Value[INT|FLOAT] | int | float] | Value) -> Value[INT|FLOAT] | Value[ERROR]`

**Description**: Sum a list of numeric values with numeric promotion.

**Usage**:
```python
total = F.sum(valueify([1, 2, 3]))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if any element is non-numeric.

#### F.min

**Signature**: `min(values: Sequence[Value[INT|FLOAT] | int | float] | Value) -> Value[INT|FLOAT] | Value[ERROR]`

**Description**: Return the minimum of a list of numeric values.

**Usage**:
```python
lowest = F.min(valueify([3, 1, 2]))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if any element is non-numeric or list is empty.

#### F.max

**Signature**: `max(values: Sequence[Value[INT|FLOAT] | int | float] | Value) -> Value[INT|FLOAT] | Value[ERROR]`

**Description**: Return the maximum of a list of numeric values.

**Usage**:
```python
highest = F.max(valueify([3, 1, 2]))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if any element is non-numeric or list is empty.

#### F.mean

**Signature**: `mean(values: Sequence[Value[INT|FLOAT] | int | float] | Value) -> Value[FLOAT] | Value[ERROR]`

**Description**: Return the mean of a list of numeric values (always FLOAT).

**Usage**:
```python
avg = F.mean(valueify([1, 2, 3]))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if any element is non-numeric or list is empty.

### Response and Metadata Ops

#### F.extract_text

**Signature**: `extract_text(resp: Value[RESPONSE] | object) -> Value[TEXT] | Value[ERROR]`

**Description**: Extract text content from a response value.

**Usage**:
```python
text = F.extract_text(response_value)
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if response content is missing or unsupported.

#### F.extract_meta

**Signature**: `extract_meta(resp: Value[RESPONSE] | object) -> Value[STRUCTURED] | Value[ERROR]`

**Description**: Extract usage/model metadata from a response value.

**Usage**:
```python
meta = F.extract_meta(response_value)
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if usage/model metadata is unavailable.

#### F.chat_complete

**Signature**: `chat_complete(client: Any, *, messages: Sequence[Mapping[str, str]] | Value[STRUCTURED], model: str | None = None, temperature: float | None = None, max_tokens: int | None = None, **kwargs: Any) -> Value[RESPONSE] | Value[ERROR]`

**Description**: Atomic transport for structured chat inputs. Executes an async OpenAI chat completion and returns the raw response object wrapped as `Value(RESPONSE)`.

**Usage**:
```python
resp = await F.chat_complete(
    client=openai_client,
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o-mini",
)
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` on API failures. Error metadata must retain HTTP
  status code information for downstream handling (e.g., `meta["http_status"]`).

#### F.with_meta

**Signature**: `with_meta(value: Value, **meta) -> Value`

**Description**: Add or overwrite metadata on a value.

**Usage**:
```python
tagged = F.with_meta(valueify("hi"), source="test")
```

**Errors**:
- If input is `Value(ERROR)`, returns the same error with merged metadata.

### Error and Control Ops

#### F.try_render

**Signature**: `try_render(template: Value[FSTRING] | str, vars: Value[STRUCTURED] | dict[str, Any]) -> Value[TEXT] | Value[ERROR]`

**Description**: Render an f-string template and return errors as values.

**Usage**:
```python
prompt = F.try_render(valueify("{x}", kind=ValueKind.FSTRING), valueify({"x": 1}))
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if template formatting fails.

#### F.unwrap_or

**Signature**: `unwrap_or(value: Value | object, default: Any) -> Value`

**Description**: Return a fallback value when the input is an error.

**Usage**:
```python
safe = F.unwrap_or(maybe_error_value, "fallback")
```

**Errors**:
- If input is `Value(ERROR)`, returns `valueify(default)` and attaches
  `meta["recovered_from"]`.

#### F.is_error

**Signature**: `is_error(value: Value | object) -> Value[STRUCTURED]`

**Description**: Return a structured flag indicating whether input is an error.

**Usage**:
```python
flag = F.is_error(maybe_error_value)
```

**Errors**:
- Never returns `Value(ERROR)`. Produces `{ "is_error": bool, "ref": ... }`.

### Collection Ops

#### F.map_values

**Signature**: `map_values(fn, values: Sequence[Value] | Value) -> list[Value]`

**Description**: Apply a function element-wise across a list/tuple batch.

**Usage**:
```python
out = F.map_values(F.strip, valueify([" a ", " b "]))
```

**Errors**:
- If an element is `Value(ERROR)`, that error is returned for that element.
- If `fn` raises or returns an error for an element, that error is preserved.
- If `values` is not list/tuple-like, returns a single `Value(ERROR)` inside the
  output list.

#### F.zip_values

**Signature**: `zip_values(*batches: Sequence[Value] | Value) -> list[tuple[Value, ...]] | Value[ERROR]`

**Description**: Zip multiple batches into tuples, validating equal lengths.

**Usage**:
```python
pairs = F.zip_values(valueify(["a", "b"]), valueify(["1", "2"]))
```

**Errors**:
- Returns `Value(ERROR)` on length mismatch.
- Does not modify contained `Value(ERROR)` elements.

#### F.stack_structured

**Signature**: `stack_structured(values: Sequence[Value[STRUCTURED] | dict | list | tuple] | Value) -> Value[STRUCTURED] | Value[ERROR]`

**Description**: Stack structured values into a single structured payload.

**Usage**:
```python
stacked = F.stack_structured([valueify({"a": 1}), valueify({"a": 2})])
```

**Errors**:
- Propagates input errors.
- Returns `Value(ERROR)` if any element is not structured.

## Design Notes

- Functions should preserve `meta` where sensible and document how conflicts
  are resolved (e.g., metadata merge strategy).
- `render()` uses f-string style `{name}` substitutions (no Jinja dependency).
- For tracing, each op should appear as a node with clear `module_name`
  (e.g., `F.render`, `F.concat`).

## Future Extensions

- Optional `F` namespace for prompt-crafting helpers (e.g., `role()`, `system()`).
- `F.trace()` helpers for debugging Value pipelines.
