# Tutorials & Cookbooks

This section collects walkthroughs and recipes. It will expand as new examples
and cookbooks are added.

## Examples

These examples provide hands-on introductions to core plait concepts:

| Example | Description |
|---------|-------------|
| [01_module.py](https://github.com/eric-tramel/plait/blob/main/examples/01_module.py) | Module basics: creating custom modules, composing hierarchies, using Parameters |
| [02_llm_pipeline.py](https://github.com/eric-tramel/plait/blob/main/examples/02_llm_pipeline.py) | LLM pipelines: sequential, parallel fan-out, fan-in synthesis patterns |
| [03_tracing.py](https://github.com/eric-tramel/plait/blob/main/examples/03_tracing.py) | Tracing: capturing execution DAGs, inspecting nodes and dependencies |
| [04_execution.py](https://github.com/eric-tramel/plait/blob/main/examples/04_execution.py) | Execution: `run()`, `bind()`, ExecutionSettings, batch processing |
| [05_optimization.py](https://github.com/eric-tramel/plait/blob/main/examples/05_optimization.py) | Optimization: train/eval modes, loss functions, backward passes |

## Cookbooks

Cookbooks demonstrate complete end-to-end workflows:

| Cookbook | Description |
|----------|-------------|
| [hallucination_detection.py](https://github.com/eric-tramel/plait/blob/main/cookbooks/hallucination_detection.py) | Training a hallucination detector with HaluBench dataset |

## Suggested Learning Path

1. Start with [01_module.py](https://github.com/eric-tramel/plait/blob/main/examples/01_module.py) to understand the basic building blocks
2. Learn LLM pipeline patterns in [02_llm_pipeline.py](https://github.com/eric-tramel/plait/blob/main/examples/02_llm_pipeline.py)
3. Understand how plait captures computation graphs with [03_tracing.py](https://github.com/eric-tramel/plait/blob/main/examples/03_tracing.py)
4. See how to execute pipelines in [04_execution.py](https://github.com/eric-tramel/plait/blob/main/examples/04_execution.py)
5. Learn optimization techniques in [05_optimization.py](https://github.com/eric-tramel/plait/blob/main/examples/05_optimization.py)
6. Apply your knowledge with the [hallucination detection cookbook](https://github.com/eric-tramel/plait/blob/main/cookbooks/hallucination_detection.py)
