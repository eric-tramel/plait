# Ordered Parameter Updates

## Problem Statement

Unlike traditional gradient-based optimization where updates are small numerical deltas, LLM parameter updates are discrete semantic changes. A mismatch between dependent parameters can break an entire pipeline:

```
[System Prompt: "Output JSON"]  →  [LLM]  →  [Validator: "Check JSON schema"]
         ↓                                            ↓
   Updated to YAML                           Still expects JSON!
                                             → PIPELINE BREAKS
```

Traditional ML: `θ_new = θ_old - lr * ∇θ` — small mismatches cause small errors.

LLM optimization: semantic changes are discrete — it either works or it doesn't.

## Solution: Topologically-Ordered Updates with Upstream Visibility

The update step must:
1. **Update in forward topological order** — upstream parameters first
2. **Provide visibility** — each update sees the new values of already-updated upstream parameters
3. **Maintain coherence** — downstream parameters can adapt to upstream changes

### Updated Flow

```
forward (through DAG) → loss → backward (reverse DAG) → step (forward DAG)
                                                              ↑
                                                    with upstream visibility
```

## Design

### Core Concept: Update Levels

Parameters are partitioned into "levels" based on their position in the DAG:

```
Level 0: Parameters with no upstream parameters (roots)
Level 1: Parameters that only depend on level 0
Level 2: Parameters that depend on levels 0-1
...
```

Parameters within the same level have no dependency relationship and can be updated **in parallel**. Levels must be processed **sequentially**.

```
     ┌─────────────────────────────────────────────────────┐
     │                    LEVEL 0                          │
     │  ┌─────────────┐         ┌─────────────┐            │
     │  │ system_spec │         │ output_fmt  │  ← parallel│
     │  └──────┬──────┘         └──────┬──────┘            │
     └─────────┼───────────────────────┼───────────────────┘
               │                       │
               ▼                       ▼
     ┌─────────────────────────────────────────────────────┐
     │                    LEVEL 1                          │
     │  ┌─────────────┐         ┌─────────────┐            │
     │  │  validator  │         │  formatter  │  ← parallel│
     │  └─────────────┘         └─────────────┘            │
     │         ↑                       ↑                   │
     │    sees updated            sees updated             │
     │    system_spec             output_fmt               │
     └─────────────────────────────────────────────────────┘
```

### API Changes

#### Optimizer Captures Records During Backward

The optimizer accumulates ForwardRecords during backward passes, then uses them for ordered updates:

```python
class Optimizer(ABC):
    def __init__(self, params: Iterable[Parameter], ...):
        self.params = list(params)
        self._records: list[ForwardRecord] = []  # Accumulated from backward()
        ...

    def zero_feedback(self) -> None:
        """Clear feedback buffers AND accumulated records.

        This resets all state for a new training batch. Must be called
        before each mini-batch iteration.
        """
        for param in self.params:
            param.zero_feedback()
        self._records.clear()  # Clear records for next batch

    def capture_record(self, record: ForwardRecord) -> None:
        """Capture a ForwardRecord during backward pass.

        Called by _propagate_backward_value() to provide graph context
        for ordered parameter updates in step().

        Args:
            record: The ForwardRecord from the forward pass.
        """
        self._records.append(record)

    @abstractmethod
    async def step(self) -> dict[str, str]:
        """Update parameters in topological order with upstream visibility."""
        pass
```

#### Backward Propagation Captures Record

In `backward.py`, `_propagate_backward_value` passes the record to the optimizer:

```python
async def _propagate_backward_value(
    feedback: Value,
    record: ForwardRecord,
    optimizer: Optimizer | None = None,
) -> None:
    """Propagate feedback and capture record for ordered updates."""

    # Capture record for topological ordering in step()
    if optimizer is not None:
        optimizer.capture_record(record)

    # ... existing backward propagation logic ...
```

### Step Implementation

```python
class SFAOptimizer(Optimizer):
    async def step(self) -> dict[str, str]:
        """Update parameters in topological order with upstream visibility.

        The algorithm:
        1. Build parameter-to-node mapping from captured records
        2. Compute update levels (topological partitioning)
        3. Process levels sequentially, parameters within level in parallel
        4. Each update receives context of upstream parameter changes

        Returns:
            Dictionary mapping parameter names to new values.
        """
        if not self._bound:
            raise RuntimeError("Optimizer not bound. Call bind(resources) first.")

        if not self._records:
            raise RuntimeError(
                "No ForwardRecords captured. Ensure module.train() is called "
                "and backward() is invoked before step()."
            )

        # Snapshot current values before any updates
        previous_values = {
            self._param_key(param): param.value
            for param in self.params
        }

        # Use first record for graph structure (all should be identical)
        graph = self._records[0].graph
        module_map = self._records[0].module_map

        # Build mappings
        param_to_nodes = self._build_param_to_node_mapping(module_map)
        node_to_params = self._build_node_to_param_mapping(module_map)

        # Compute update levels
        levels = self._compute_update_levels(graph, param_to_nodes)

        # Track applied updates for visibility
        applied_updates: dict[str, str] = {}

        # Process levels sequentially
        for level in levels:
            # Filter to params that need updating
            params_to_update = [
                p for p in level
                if p.requires_grad and p._feedback_buffer
            ]

            if not params_to_update:
                continue

            # Update all params in this level in parallel
            level_results = await asyncio.gather(*[
                self._update_parameter_with_context(
                    param=param,
                    previous_values=previous_values,
                    applied_updates=applied_updates,
                    upstream_params=self._get_upstream_params(
                        param, param_to_nodes, node_to_params, graph
                    ),
                )
                for param in params_to_update
            ])

            # Apply updates and record them
            for param, new_value in level_results:
                param.apply_update(new_value)
                key = self._param_key(param)
                applied_updates[key] = new_value

        self._step_count += 1
        return applied_updates

    def _param_key(self, param: Parameter) -> str:
        """Get a consistent key for a parameter."""
        return param._name or str(id(param))
```

### Computing Update Levels

```python
def _compute_update_levels(
    self,
    graph: InferenceGraph,
    param_to_nodes: dict[str, set[str]],
) -> list[list[Parameter]]:
    """Partition parameters into topological levels.

    Level 0 contains parameters with no upstream parameters.
    Level N contains parameters whose upstream params are all in levels < N.

    Args:
        graph: The inference graph.
        param_to_nodes: Mapping from param key to node IDs containing it.

    Returns:
        List of levels, where each level is a list of Parameters
        that can be updated in parallel.
    """
    # Get topological order of nodes
    topo_order = graph.topological_order()
    node_to_topo_index = {nid: i for i, nid in enumerate(topo_order)}

    # Compute "earliest" topological index for each parameter
    # (a param might appear in multiple nodes; use the earliest)
    param_topo_index: dict[str, int] = {}
    for param in self.params:
        key = self._param_key(param)
        nodes = param_to_nodes.get(key, set())
        if nodes:
            param_topo_index[key] = min(node_to_topo_index[n] for n in nodes)
        else:
            # Parameter not in graph (e.g., not used in this forward pass)
            param_topo_index[key] = -1

    # Group parameters by their topological index
    index_to_params: dict[int, list[Parameter]] = {}
    for param in self.params:
        key = self._param_key(param)
        idx = param_topo_index[key]
        if idx >= 0:  # Skip params not in graph
            index_to_params.setdefault(idx, []).append(param)

    # Convert to ordered list of levels
    # Parameters at same topo index are in same level (can parallelize)
    levels = [
        params
        for idx in sorted(index_to_params.keys())
        for params in [index_to_params[idx]]
    ]

    return levels
```

### Building Mappings

```python
def _build_param_to_node_mapping(
    self,
    module_map: dict[str, Module],
) -> dict[str, set[str]]:
    """Map parameter keys to the nodes containing them.

    Args:
        module_map: Mapping from node_id to module instance.

    Returns:
        Dict mapping param key to set of node IDs where it appears.
    """
    mapping: dict[str, set[str]] = {}

    for node_id, module in module_map.items():
        for name, param in module.named_parameters():
            key = self._param_key(param)
            mapping.setdefault(key, set()).add(node_id)

    return mapping

def _build_node_to_param_mapping(
    self,
    module_map: dict[str, Module],
) -> dict[str, list[Parameter]]:
    """Map node IDs to parameters in that node's module.

    Args:
        module_map: Mapping from node_id to module instance.

    Returns:
        Dict mapping node_id to list of Parameters in that module.
    """
    mapping: dict[str, list[Parameter]] = {}

    for node_id, module in module_map.items():
        params = list(module.parameters())
        if params:
            mapping[node_id] = params

    return mapping

def _get_upstream_params(
    self,
    param: Parameter,
    param_to_nodes: dict[str, set[str]],
    node_to_params: dict[str, list[Parameter]],
    graph: InferenceGraph,
) -> list[Parameter]:
    """Get all parameters upstream of this parameter.

    Args:
        param: The parameter to find upstream params for.
        param_to_nodes: Mapping from param key to node IDs.
        node_to_params: Mapping from node ID to parameters.
        graph: The inference graph.

    Returns:
        List of Parameters that are upstream (ancestors) of this param.
    """
    key = self._param_key(param)
    nodes = param_to_nodes.get(key, set())

    if not nodes:
        return []

    # Get all ancestor nodes
    ancestor_nodes: set[str] = set()
    for node_id in nodes:
        ancestor_nodes.update(graph.ancestors(node_id))

    # Collect parameters from ancestor nodes
    upstream_params: list[Parameter] = []
    seen: set[str] = set()

    for node_id in ancestor_nodes:
        for p in node_to_params.get(node_id, []):
            p_key = self._param_key(p)
            if p_key not in seen and p_key != key:
                seen.add(p_key)
                upstream_params.append(p)

    return upstream_params
```

### Generating Updates with Upstream Context

```python
async def _update_parameter_with_context(
    self,
    param: Parameter,
    previous_values: dict[str, str],
    applied_updates: dict[str, str],
    upstream_params: list[Parameter],
) -> tuple[Parameter, str]:
    """Generate an update for a parameter with upstream visibility.

    Args:
        param: The parameter to update.
        previous_values: Snapshot of all param values before step().
        applied_updates: Updates already applied in earlier levels.
        upstream_params: Parameters upstream of this one.

    Returns:
        Tuple of (parameter, new_value).
    """
    # Aggregate feedback for this parameter
    aggregated = await self._aggregate_feedback(param)

    # Build upstream context showing what changed
    upstream_context = self._build_upstream_context(
        upstream_params,
        previous_values,
        applied_updates,
    )

    # Generate update with context
    new_value = await self._generate_update_with_context(
        param,
        aggregated,
        upstream_context,
    )

    return (param, new_value)

async def _aggregate_feedback(self, param: Parameter) -> str:
    """Combine all feedback items into one coherent summary.

    Args:
        param: The parameter whose feedback should be aggregated.

    Returns:
        Aggregated feedback summary as a string.
    """
    feedbacks = param._feedback_buffer

    if len(feedbacks) == 1:
        return feedbacks[0]

    feedback_items = "\n".join(
        f"<feedback index=\"{i + 1}\">{fb}</feedback>"
        for i, fb in enumerate(feedbacks)
    )

    prompt = f"""<task>Aggregate feedback for a parameter</task>

<parameter name="{param._name}">
<description>{param.description}</description>
<current-value>{param.value}</current-value>
</parameter>

<feedback-items count="{len(feedbacks)}">
{feedback_items}
</feedback-items>

<instructions>
Synthesize the feedback items into a single coherent summary that:
- Identifies common themes across feedback items
- Prioritizes the most impactful suggestions
- Notes and resolves any conflicting feedback
- Provides specific, actionable recommendations
</instructions>

<output-format>
Provide a single summary paragraph. No bullet points, no numbered lists.
</output-format>"""

    return await self.aggregator(prompt)

def _build_upstream_context(
    self,
    upstream_params: list[Parameter],
    previous_values: dict[str, str],
    applied_updates: dict[str, str],
) -> str:
    """Build context string showing upstream parameter changes.

    Args:
        upstream_params: Parameters that are upstream in the graph.
        previous_values: Values before any updates.
        applied_updates: Updates already applied.

    Returns:
        Formatted string describing upstream changes, or empty if none.
    """
    changed_params = []

    for param in upstream_params:
        key = self._param_key(param)
        if key in applied_updates:
            changed_params.append({
                "name": param._name or key,
                "description": param.description,
                "previous": previous_values.get(key, ""),
                "updated": applied_updates[key],
            })

    if not changed_params:
        return ""

    param_blocks = []
    for p in changed_params:
        param_blocks.append(f"""<upstream-parameter name="{p['name']}">
<description>{p['description']}</description>
<previous-value>{p['previous']}</previous-value>
<updated-value>{p['updated']}</updated-value>
</upstream-parameter>""")

    return f"""<upstream-updates>
<note>
The following parameters appear earlier in the pipeline and have already
been updated in this optimization step. Your update should maintain
consistency with these changes.
</note>
{chr(10).join(param_blocks)}
</upstream-updates>"""

async def _generate_update_with_context(
    self,
    param: Parameter,
    aggregated: str,
    upstream_context: str,
) -> str:
    """Generate improved parameter value with upstream context.

    Args:
        param: The parameter to update.
        aggregated: Aggregated feedback for this parameter.
        upstream_context: Context about upstream parameter changes.

    Returns:
        The new parameter value.
    """
    prompt = f"""<task>Update a parameter based on feedback</task>

<parameter name="{param._name}">
<description>{param.description}</description>
<current-value>{param.value}</current-value>
</parameter>

<aggregated-feedback>
{aggregated}
</aggregated-feedback>

{upstream_context}

<update-guidelines>
<conservatism>{self.conservatism:.1f}</conservatism>
<conservatism-scale>0 = aggressive rewrites, 1 = minimal changes</conservatism-scale>
<instructions>
- Address the key points in the feedback
- Preserve aspects that are working well
- Make changes proportional to the conservatism level
- Maintain consistency with any upstream parameter changes
</instructions>
</update-guidelines>

<output-format>
Return ONLY the new parameter value. No explanations, no markdown, no quotes.
</output-format>"""

    return await self.updater(prompt)
```

## Example: Coordinated Update

Consider a pipeline with a format specification and a validator:

```python
class FormattedPipeline(Module):
    def __init__(self):
        super().__init__()
        self.format_spec = Parameter(
            value="Output as JSON with keys: name, age, city",
            description="Specifies the output format for the LLM",
        )
        self.validator_rules = Parameter(
            value="Verify output is valid JSON with required keys",
            description="Validation rules that must match the format spec",
        )
        self.llm = LLMInference(alias="main")
        self.validator = LLMInference(alias="validator")

    def forward(self, query: str) -> str:
        prompt = f"{self.format_spec}\n\nQuery: {query}"
        response = self.llm(prompt)
        validated = self.validator(f"{self.validator_rules}\n\n{response}")
        return validated
```

### Without Ordered Updates (Current)

```
Step 1: Update format_spec
  Value: "Users prefer YAML format"
  → Updated to: "Output as YAML with keys: name, age, city"

Step 2: Update validator_rules (no visibility!)
  Value: "Validation is too strict"
  → Updated to: "Verify output is valid JSON, allow extra keys"

RESULT: format_spec expects YAML, validator expects JSON → BROKEN
```

### With Ordered Updates (New)

```
Step 1 (Level 0): Update format_spec
  Value: "Users prefer YAML format"
  → Updated to: "Output as YAML with keys: name, age, city"

Step 2 (Level 1): Update validator_rules (sees upstream change!)
  Value: "Validation is too strict"
  Upstream Context:
    format_spec changed from JSON to YAML
  → Updated to: "Verify output is valid YAML, allow extra keys"

RESULT: Both parameters are consistent → WORKS
```

## Handling Edge Cases

### 1. No Records Captured

If `step()` is called without any captured records, this is an error. The optimizer should only be used during training mode, which always produces records:

```python
if not self._records:
    raise RuntimeError(
        "No ForwardRecords captured. Ensure module.train() is called "
        "and backward() is invoked before step()."
    )
```

### 2. Same Parameter in Multiple Nodes

A parameter might be used in multiple nodes (e.g., shared system prompt used by two LLM calls). The behavior:

1. **During backward**: The parameter accumulates feedback from ALL nodes that use it (each backward pass through a node contributes feedback)
2. **During step**: The parameter is updated ONCE at its earliest topological position, using all accumulated feedback
3. **Visibility**: When later nodes are processed, other parameters in those nodes see the already-updated shared parameter

```python
# Update once at earliest position
param_topo_index[key] = min(node_to_topo_index[n] for n in nodes)

# All accumulated feedback (from all nodes) is used for this single update
aggregated = await self._aggregate_feedback(param)  # Contains feedback from all usages
```

This ensures:
- Coherent update that considers all usages
- No conflicting updates from processing the same parameter multiple times
- Downstream parameters see the updated value

### 3. No Upstream Parameters Changed

If no upstream parameters received updates, the context is empty and the update proceeds as before:

```python
if not changed_params:
    return ""
```

### 4. Circular Dependencies

The graph is a DAG (enforced by tracing), so circular dependencies are impossible. `topological_order()` raises `ValueError` if a cycle is detected.

## Performance Considerations

### Parallelism Within Levels

Parameters at the same topological level have no dependencies and can be updated concurrently:

```python
level_results = await asyncio.gather(*[
    self._update_parameter_with_context(...)
    for param in params_to_update
])
```

### Sequential Overhead

The overhead of processing levels sequentially is minimal compared to the LLM calls for aggregation and update generation. Each level requires waiting for all updates to complete before proceeding.

### Caching Mappings

For training loops with the same graph structure, mappings could be cached:

```python
# Future optimization
if self._cached_graph_hash == graph.compute_hash():
    # Reuse cached mappings
    ...
```

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Update order | Arbitrary (registration order) | Topological (forward DAG) |
| Upstream visibility | None | Full context of changes |
| Consistency | Not guaranteed | Enforced by design |
| Parallelism | All params parallel | Parallel within levels |
| Record capture | N/A | `capture_record()` called by backward |
| Record clearing | N/A | `zero_feedback()` clears records |
| No records | Silent no-op | RuntimeError (invalid usage) |
| Prompt format | Markdown/headers | XML tags for content demarcation |

The key insight: **LLM parameter updates are discrete semantic changes, not smooth gradients. Coherent systems require coordinated updates with visibility of upstream changes.**
