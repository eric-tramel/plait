"""Unit tests for the Proxy class."""

from typing import Any

from inf_engine.tracing.proxy import Proxy
from inf_engine.tracing.tracer import Tracer


class MockingTracer(Tracer):
    """A tracer subclass that implements record_getitem for testing.

    Extends the real Tracer class to provide a working implementation
    of record_getitem for Proxy tests, since the real implementation
    is deferred to a future PR.
    """

    def __init__(self) -> None:
        super().__init__()
        self.recorded_getitems: list[tuple[Proxy, Any]] = []
        self._getitem_counter: int = 0

    def record_getitem(self, proxy: Proxy, key: Any) -> Proxy:
        """Record a getitem operation and return a new proxy."""
        self.recorded_getitems.append((proxy, key))
        self._getitem_counter += 1
        return Proxy(
            node_id=f"getitem_{self._getitem_counter}",
            tracer=self,
        )


class TestProxyCreation:
    """Tests for Proxy instantiation."""

    def test_proxy_creation_basic(self) -> None:
        """Proxy can be created with node_id and tracer."""
        tracer = MockingTracer()
        proxy = Proxy(node_id="test_node", tracer=tracer)

        assert proxy.node_id == "test_node"
        assert proxy.tracer is tracer

    def test_proxy_creation_with_defaults(self) -> None:
        """Proxy has correct default values."""
        tracer = MockingTracer()
        proxy = Proxy(node_id="test_node", tracer=tracer)

        assert proxy.output_index == 0
        assert proxy._metadata == {}

    def test_proxy_creation_with_output_index(self) -> None:
        """Proxy can be created with a custom output_index."""
        tracer = MockingTracer()
        proxy = Proxy(node_id="test_node", tracer=tracer, output_index=2)

        assert proxy.output_index == 2

    def test_proxy_creation_with_metadata(self) -> None:
        """Proxy can be created with custom metadata."""
        tracer = MockingTracer()
        metadata = {"key": "value", "count": 42}
        proxy = Proxy(node_id="test_node", tracer=tracer, _metadata=metadata)

        assert proxy._metadata == {"key": "value", "count": 42}

    def test_proxy_metadata_is_independent(self) -> None:
        """Each proxy has its own metadata dictionary."""
        tracer = MockingTracer()
        proxy1 = Proxy(node_id="node1", tracer=tracer)
        proxy2 = Proxy(node_id="node2", tracer=tracer)

        proxy1._metadata["key"] = "value1"
        proxy2._metadata["key"] = "value2"

        assert proxy1._metadata["key"] == "value1"
        assert proxy2._metadata["key"] == "value2"

    def test_proxy_creation_with_real_tracer(self) -> None:
        """Proxy works with the real Tracer class."""
        tracer = Tracer()
        proxy = Proxy(node_id="test_node", tracer=tracer)

        assert proxy.node_id == "test_node"
        assert proxy.tracer is tracer


class TestProxyRepr:
    """Tests for Proxy string representation."""

    def test_proxy_repr_format(self) -> None:
        """repr returns the expected format."""
        tracer = MockingTracer()
        proxy = Proxy(node_id="LLMInference_1", tracer=tracer)

        assert repr(proxy) == "Proxy(LLMInference_1)"

    def test_proxy_repr_with_different_ids(self) -> None:
        """repr works with various node_id formats."""
        tracer = MockingTracer()

        test_cases = [
            "simple",
            "Module_123",
            "input:prompt",
            "nested.path.node",
            "",  # edge case: empty node_id
        ]

        for node_id in test_cases:
            proxy = Proxy(node_id=node_id, tracer=tracer)
            assert repr(proxy) == f"Proxy({node_id})"

    def test_proxy_str_same_as_repr(self) -> None:
        """str() uses the same representation as repr()."""
        tracer = MockingTracer()
        proxy = Proxy(node_id="test_node", tracer=tracer)

        # Dataclass default behavior: str uses repr
        assert str(proxy) == repr(proxy)


class TestProxyGetitem:
    """Tests for Proxy.__getitem__()."""

    def test_proxy_getitem_returns_proxy(self) -> None:
        """__getitem__ returns a new Proxy."""
        tracer = MockingTracer()
        proxy = Proxy(node_id="dict_output", tracer=tracer)

        result = proxy["key"]

        assert isinstance(result, Proxy)

    def test_proxy_getitem_delegates_to_tracer(self) -> None:
        """__getitem__ calls tracer.record_getitem()."""
        tracer = MockingTracer()
        proxy = Proxy(node_id="dict_output", tracer=tracer)

        proxy["my_key"]

        assert len(tracer.recorded_getitems) == 1
        recorded_proxy, recorded_key = tracer.recorded_getitems[0]
        assert recorded_proxy is proxy
        assert recorded_key == "my_key"

    def test_proxy_getitem_with_string_key(self) -> None:
        """__getitem__ works with string keys."""
        tracer = MockingTracer()
        proxy = Proxy(node_id="result", tracer=tracer)

        result = proxy["output"]

        assert result.node_id == "getitem_1"

    def test_proxy_getitem_with_integer_index(self) -> None:
        """__getitem__ works with integer indices."""
        tracer = MockingTracer()
        proxy = Proxy(node_id="list_output", tracer=tracer)

        result = proxy[0]

        assert isinstance(result, Proxy)
        _, recorded_key = tracer.recorded_getitems[0]
        assert recorded_key == 0

    def test_proxy_getitem_multiple_accesses(self) -> None:
        """Multiple __getitem__ calls create separate nodes."""
        tracer = MockingTracer()
        proxy = Proxy(node_id="data", tracer=tracer)

        result1 = proxy["first"]
        result2 = proxy["second"]
        result3 = proxy[0]

        assert result1.node_id == "getitem_1"
        assert result2.node_id == "getitem_2"
        assert result3.node_id == "getitem_3"
        assert len(tracer.recorded_getitems) == 3

    def test_proxy_getitem_chaining(self) -> None:
        """__getitem__ can be chained for nested access."""
        tracer = MockingTracer()
        proxy = Proxy(node_id="nested_dict", tracer=tracer)

        # Simulate nested["outer"]["inner"]
        outer = proxy["outer"]
        inner = outer["inner"]

        assert len(tracer.recorded_getitems) == 2
        assert inner.node_id == "getitem_2"


class TestProxyIter:
    """Tests for Proxy.__iter__()."""

    def test_proxy_iter_returns_iterator(self) -> None:
        """__iter__ returns an iterator."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", [1, 2, 3])

        result = iter(input_proxy)

        # Should be an iterator
        assert hasattr(result, "__next__")

    def test_proxy_iter_yields_proxy(self) -> None:
        """__iter__ yields Proxy objects."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", [1, 2, 3])

        items = list(input_proxy)

        assert len(items) == 1
        assert isinstance(items[0], Proxy)

    def test_proxy_iter_creates_node(self) -> None:
        """__iter__ creates an iter node in the graph."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", [1, 2, 3])

        list(input_proxy)  # Consume the iterator

        assert "iter_1" in tracer.nodes
        assert tracer.nodes["iter_1"].dependencies == ["input:data"]

    def test_proxy_iter_for_loop(self) -> None:
        """__iter__ works with for loops."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", ["a", "b"])

        collected = []
        for item in input_proxy:
            collected.append(item)

        assert len(collected) == 1
        assert isinstance(collected[0], Proxy)


class TestProxyKeys:
    """Tests for Proxy.keys()."""

    def test_proxy_keys_returns_proxy(self) -> None:
        """keys() returns a Proxy."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        result = input_proxy.keys()

        assert isinstance(result, Proxy)

    def test_proxy_keys_creates_node(self) -> None:
        """keys() creates a method node in the graph."""
        from inf_engine.tracing.tracer import MethodOp

        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        input_proxy.keys()

        assert "method_1" in tracer.nodes
        node = tracer.nodes["method_1"]
        assert node.dependencies == ["input:data"]
        assert isinstance(node.module, MethodOp)
        assert node.module.method == "keys"


class TestProxyValues:
    """Tests for Proxy.values()."""

    def test_proxy_values_returns_proxy(self) -> None:
        """values() returns a Proxy."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        result = input_proxy.values()

        assert isinstance(result, Proxy)

    def test_proxy_values_creates_node(self) -> None:
        """values() creates a method node in the graph."""
        from inf_engine.tracing.tracer import MethodOp

        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        input_proxy.values()

        assert "method_1" in tracer.nodes
        node = tracer.nodes["method_1"]
        assert node.dependencies == ["input:data"]
        assert isinstance(node.module, MethodOp)
        assert node.module.method == "values"


class TestProxyItems:
    """Tests for Proxy.items()."""

    def test_proxy_items_returns_proxy(self) -> None:
        """items() returns a Proxy."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        result = input_proxy.items()

        assert isinstance(result, Proxy)

    def test_proxy_items_creates_node(self) -> None:
        """items() creates a method node in the graph."""
        from inf_engine.tracing.tracer import MethodOp

        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        input_proxy.items()

        assert "method_1" in tracer.nodes
        node = tracer.nodes["method_1"]
        assert node.dependencies == ["input:data"]
        assert isinstance(node.module, MethodOp)
        assert node.module.method == "items"


class TestProxyOperationsChaining:
    """Tests for chaining Proxy operations."""

    def test_keys_then_iter(self) -> None:
        """Can chain keys() with iteration."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1, "b": 2})

        keys_proxy = input_proxy.keys()
        items = list(keys_proxy)

        assert len(items) == 1
        assert isinstance(items[0], Proxy)
        # Should have 3 nodes: input, keys, iter
        assert len(tracer.nodes) == 3

    def test_getitem_then_keys(self) -> None:
        """Can chain getitem with keys()."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"nested": {"a": 1}})

        nested = input_proxy["nested"]
        keys = nested.keys()

        assert isinstance(keys, Proxy)
        # Should have 3 nodes: input, getitem, keys
        assert len(tracer.nodes) == 3

    def test_items_iteration_pattern(self) -> None:
        """Can use items() in a for loop pattern."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1, "b": 2})

        collected = []
        for item in input_proxy.items():
            collected.append(item)

        assert len(collected) == 1
        # Should have 3 nodes: input, items, iter
        assert len(tracer.nodes) == 3


class TestProxyEquality:
    """Tests for Proxy equality comparison."""

    def test_proxy_equality_same_fields(self) -> None:
        """Proxies with same fields are equal."""
        tracer = MockingTracer()
        proxy1 = Proxy(node_id="test", tracer=tracer, output_index=0)
        proxy2 = Proxy(node_id="test", tracer=tracer, output_index=0)

        assert proxy1 == proxy2

    def test_proxy_inequality_different_node_id(self) -> None:
        """Proxies with different node_ids are not equal."""
        tracer = MockingTracer()
        proxy1 = Proxy(node_id="node1", tracer=tracer)
        proxy2 = Proxy(node_id="node2", tracer=tracer)

        assert proxy1 != proxy2

    def test_proxy_inequality_different_output_index(self) -> None:
        """Proxies with different output_index are not equal."""
        tracer = MockingTracer()
        proxy1 = Proxy(node_id="test", tracer=tracer, output_index=0)
        proxy2 = Proxy(node_id="test", tracer=tracer, output_index=1)

        assert proxy1 != proxy2

    def test_proxy_inequality_different_tracer(self) -> None:
        """Proxies with different tracers are not equal."""
        tracer1 = MockingTracer()
        tracer2 = MockingTracer()
        proxy1 = Proxy(node_id="test", tracer=tracer1)
        proxy2 = Proxy(node_id="test", tracer=tracer2)

        assert proxy1 != proxy2
