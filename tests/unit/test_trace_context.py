"""Unit tests for the trace context infrastructure."""

from plait.tracing.context import get_trace_context, trace_context
from plait.tracing.tracer import Tracer


class NamedTracer(Tracer):
    """A tracer subclass with additional test-specific attributes.

    Extends the real Tracer class to add a name attribute for testing
    context manager behavior with distinguishable instances.
    """

    def __init__(self, name: str = "mock") -> None:
        super().__init__()
        self.name = name


class TestGetTraceContext:
    """Tests for get_trace_context()."""

    def test_context_default_none(self) -> None:
        """No context is set by default."""
        assert get_trace_context() is None

    def test_context_returns_none_outside_block(self) -> None:
        """Context returns None when not inside a trace_context block."""
        # Explicitly verify the default state
        ctx = get_trace_context()
        assert ctx is None


class TestTraceContext:
    """Tests for the trace_context context manager."""

    def test_context_set_and_get(self) -> None:
        """Context is available inside the block."""
        tracer = NamedTracer()
        with trace_context(tracer):
            ctx = get_trace_context()
            assert ctx is tracer

    def test_context_cleared_after(self) -> None:
        """Context is None after exiting the block."""
        tracer = NamedTracer()
        with trace_context(tracer):
            assert get_trace_context() is tracer
        # After exiting, context should be cleared
        assert get_trace_context() is None

    def test_context_yields_tracer(self) -> None:
        """trace_context yields the tracer passed to it."""
        tracer = NamedTracer("test_tracer")
        with trace_context(tracer) as ctx:
            assert ctx is tracer
            assert ctx.name == "test_tracer"

    def test_nested_contexts(self) -> None:
        """Nested contexts work correctly with proper restoration."""
        outer_tracer = NamedTracer("outer")
        inner_tracer = NamedTracer("inner")

        assert get_trace_context() is None

        with trace_context(outer_tracer):
            assert get_trace_context() is outer_tracer

            with trace_context(inner_tracer):
                assert get_trace_context() is inner_tracer

            # After inner exits, outer should be restored
            assert get_trace_context() is outer_tracer

        # After outer exits, should be None
        assert get_trace_context() is None

    def test_context_cleared_on_exception(self) -> None:
        """Context is properly cleared even when an exception occurs."""
        tracer = NamedTracer()

        try:
            with trace_context(tracer):
                assert get_trace_context() is tracer
                raise ValueError("test exception")
        except ValueError:
            pass

        # Context should still be cleared
        assert get_trace_context() is None

    def test_nested_context_restored_on_exception(self) -> None:
        """Nested contexts restore properly even with exceptions."""
        outer_tracer = NamedTracer("outer")
        inner_tracer = NamedTracer("inner")

        with trace_context(outer_tracer):
            try:
                with trace_context(inner_tracer):
                    assert get_trace_context() is inner_tracer
                    raise ValueError("test exception")
            except ValueError:
                pass

            # Outer should be restored after inner raises
            assert get_trace_context() is outer_tracer

        assert get_trace_context() is None

    def test_multiple_sequential_contexts(self) -> None:
        """Multiple sequential context managers work correctly."""
        tracer1 = NamedTracer("first")
        tracer2 = NamedTracer("second")
        tracer3 = NamedTracer("third")

        with trace_context(tracer1):
            assert get_trace_context() is tracer1

        assert get_trace_context() is None

        with trace_context(tracer2):
            assert get_trace_context() is tracer2

        assert get_trace_context() is None

        with trace_context(tracer3):
            assert get_trace_context() is tracer3

        assert get_trace_context() is None

    def test_same_tracer_can_be_used_multiple_times(self) -> None:
        """The same tracer instance can be used in multiple context blocks."""
        tracer = NamedTracer()

        with trace_context(tracer):
            assert get_trace_context() is tracer

        assert get_trace_context() is None

        with trace_context(tracer):
            assert get_trace_context() is tracer

        assert get_trace_context() is None

    def test_with_real_tracer(self) -> None:
        """Works with the real Tracer class directly."""
        tracer = Tracer()

        with trace_context(tracer):
            ctx = get_trace_context()
            assert ctx is tracer
            assert ctx.nodes == {}

        assert get_trace_context() is None


class TestTraceContextIntegration:
    """Integration tests for trace context with module-like behavior."""

    def test_simulates_module_call_behavior(self) -> None:
        """Simulates how Module.__call__ will use the context."""
        tracer = NamedTracer()
        call_recorded = False

        def mock_module_call() -> str:
            nonlocal call_recorded
            ctx = get_trace_context()
            if ctx is not None:
                # During tracing, record the call
                call_recorded = True
                return "traced_result"
            else:
                # During execution, run forward
                return "actual_result"

        # Outside trace context - executes normally
        result = mock_module_call()
        assert result == "actual_result"
        assert not call_recorded

        # Inside trace context - records the call
        with trace_context(tracer):
            result = mock_module_call()
            assert result == "traced_result"
            assert call_recorded
