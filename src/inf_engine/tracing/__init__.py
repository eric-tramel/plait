"""Tracing and DAG capture for inference modules.

This subpackage provides the tracing infrastructure for capturing execution
graphs from eager-mode Python code, similar to torch.fx.

Note:
    The trace context functions (`get_trace_context`, `trace_context`) are
    internal infrastructure used by the framework. They are not part of the
    public API exported from the top-level `inf_engine` package.

    Internal usage:
        - `InferenceModule.__call__` checks context to record calls during tracing
        - `Tracer.trace()` establishes the context for graph capture
        - `branch` decorator checks context to handle conditional tracing

    Users typically interact with higher-level APIs like `Tracer`, `run()`,
    and the `@branch` decorator rather than these context functions directly.
"""

from inf_engine.tracing.context import get_trace_context, trace_context
from inf_engine.tracing.proxy import Proxy

__all__ = ["Proxy", "get_trace_context", "trace_context"]
