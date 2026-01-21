# Optimization

This document describes plait’s **value-only** optimization workflow. The API
mirrors PyTorch’s pattern:

```
forward → loss() → loss.backward() → optimizer.step()
```

Key differences from classic autograd:
- Feedback is **textual** and stored in `Value` payloads.
- Tape ids are stored in `Value.meta["_tape_ids"]`.
- The active optimizer is discovered automatically; `backward()` does not take
  an optimizer argument.

## Core Types

### Value

`plait.values.Value` is the single canonical container for forward outputs and
loss feedback.

- `kind`: `ValueKind` discriminator (TEXT, STRUCTURED, etc.)
- `payload`: underlying data
- `ref`: optional graph node id
- `meta`: metadata; tape ids live in `meta["_tape_ids"]`

Feedback/loss Values use `ValueKind.STRUCTURED` with canonical payload shape
`list[list[str]]`. Optional numeric scores live in `meta["score"]`.

```python
from plait.values import Value, ValueKind

loss = Value(ValueKind.STRUCTURED, [["Too verbose"]], meta={"score": 0.2})
```

Compute loss inside the traced graph so the loss Value carries tape ids, then
call `loss.backward()`.

`Value.backward(grad=None, retain_graph=False)` collects tape ids from the
Value (or nested structures) and propagates feedback through each recorded
forward pass. It raises if no tape ids are attached. When calling backward()
on a container of Values, you may pass per-record grads as a list/tuple
matched to the collected tape ids, or as a dict keyed by tape id.

### ForwardRecord (Tape)

`ForwardRecord` captures a forward pass (graph + node inputs/outputs). Records
are stored in a registry and referenced by tape ids in `Value.meta`.

- Training mode (`module.train()`) or `run(..., record=True)` attaches tape ids
  to outputs.
- `Value.backward()` uses the tape ids to resolve records.

## Loss Modules

Losses are `Module` instances (see `LossModule`) that return `Value` feedback
and can participate in tracing. The public loss factories live in
`plait.optimization.loss`:

- `VerifierLoss`
- `HumanFeedbackLoss`
- `LLMRubricLoss` / `HumanRubricLoss`
- `LLMPreferenceLoss` / `HumanPreferenceLoss`
- `LLMRankingLoss` / `HumanRankingLoss`
- `CompositeLoss`

All loss modules normalize feedback into `list[list[str]]` payloads and may
attach `meta["score"]` for numeric scoring.

Note: LLM-based losses (e.g., `LLMRubricLoss`) still require `loss_fn.bind(resources)`
to bind their internal judge wrappers.

## TrainingStep (Recommended)

Use `TrainingStep` to compose a model and loss into a single traced step so
`loss.backward()` works directly:

```python
step = TrainingStep(module, loss_fn)
step.train()

loss = await step(input, target)
await loss.backward()

await optimizer.step()
step.eval()
```

## Optimizer

Optimizers follow the PyTorch pattern and are configured upfront with
parameters:

```python
optimizer = SFAOptimizer(module.parameters(), conservatism=0.7).bind(resources)
```

The optimizer is automatically registered as the **active optimizer** on
instantiation, `bind()`, and `zero_feedback()`. Typical training loops do not
need any extra activation calls.

`optimizer.step()` applies updates based on accumulated feedback.

## Training Loop (Single‑Sample)

```python
step = TrainingStep(module, loss_fn)
step.train()
optimizer.zero_feedback()

loss = await step(input, target)
await loss.backward()

await optimizer.step()
step.eval()
```

## Batch Training

```python
step = TrainingStep(module, loss_fn)
step.train()
optimizer.zero_feedback()

losses = await asyncio.gather(
    *[step(x, target=t) for x, t in zip(batch_inputs, targets, strict=True)]
)
await asyncio.gather(*[loss.backward() for loss in losses])

await optimizer.step()
step.eval()
```

Advanced: if you already have outputs with tape ids, you can still call
`Value.backward(outputs, grad=combined_loss)` for aggregated feedback, or pass
per-record grads as `grad=[loss_a, loss_b]` (list/tuple) or
`grad={tape_id: loss_val}` (dict). If you manage multiple optimizers in the
same process, ensure you call `optimizer.zero_feedback()` on the one you want
active before backward.

## Backward Context

`Module.backward()` receives a `BackwardContext` with the forward inputs,
outputs, graph, and downstream feedback. If the active optimizer was created
with `reasoning_model=...`, `ctx.reason()` is available for LLM-powered
feedback generation.
