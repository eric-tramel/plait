"""Unit tests for functional error propagation and resolution."""

import inf_engine.functional as F
from inf_engine.values import Value, ValueKind, valueify


class TestErrorResolution:
    """Tests for deterministic error resolution."""

    def test_error_resolution_order_and_causes(self) -> None:
        """Primary error is first positional; others become causes."""
        err1 = Value(
            ValueKind.ERROR, ValueError("first"), ref="ref1", meta={"tag": "p"}
        )
        err2 = Value(ValueKind.ERROR, ValueError("second"), ref="ref2")
        err3 = Value(ValueKind.ERROR, ValueError("third"), ref="ref3")

        result = F.concat(err1, "ok", err2, sep=err3)

        assert result.kind == ValueKind.ERROR
        assert result.payload is err1.payload
        assert result.meta["tag"] == "p"
        assert result.meta["causes"] == [
            {"ref": "ref2", "message": "second"},
            {"ref": "ref3", "message": "third"},
        ]

    def test_error_resolution_dict_order(self) -> None:
        """Errors inside dicts are resolved by sorted key order."""
        err_a = Value(ValueKind.ERROR, ValueError("a"), ref="ref-a")
        err_b = Value(ValueKind.ERROR, ValueError("b"), ref="ref-b")
        vars_value = {"b": err_b, "a": err_a}

        result = F.render(valueify("{x}", kind=ValueKind.FSTRING), vars_value)

        assert result.kind == ValueKind.ERROR
        assert result.payload is err_a.payload
        assert result.meta["causes"] == [{"ref": "ref-b", "message": "b"}]

    def test_select_error_precedence_over_default(self) -> None:
        """select returns error input even when default is provided."""
        err = Value(ValueKind.ERROR, ValueError("boom"), ref="err")

        result = F.select(err, "a", default="fallback")

        assert result is err
