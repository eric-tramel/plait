"""Unit tests for resource configuration.

This module tests EndpointConfig, ResourceConfig, and their preset variants.
Tests are consolidated using parametrize to reduce redundancy while
maintaining coverage of critical functionality.
"""

import pytest

from plait.resources.config import (
    AnthropicEndpointConfig,
    EndpointConfig,
    NvidiaBuildEndpointConfig,
    OpenAIEndpointConfig,
    ResourceConfig,
)


class TestEndpointConfig:
    """Tests for EndpointConfig creation and configuration."""

    def test_creation_with_all_fields(self) -> None:
        """EndpointConfig correctly stores all configuration fields."""
        config = EndpointConfig(
            provider_api="vllm",
            model="mistral-7b",
            base_url="http://localhost:8000",
            api_key="test-key",
            max_concurrent=50,
            rate_limit=100.0,
            max_retries=5,
            retry_delay=2.0,
            timeout=600.0,
            input_cost_per_1m=1.0,
            output_cost_per_1m=2.0,
        )

        assert config.provider_api == "vllm"
        assert config.model == "mistral-7b"
        assert config.base_url == "http://localhost:8000"
        assert config.api_key == "test-key"
        assert config.max_concurrent == 50
        assert config.rate_limit == 100.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.timeout == 600.0
        assert config.input_cost_per_1m == 1.0
        assert config.output_cost_per_1m == 2.0

    @pytest.mark.parametrize(
        "field,default",
        [
            ("base_url", None),
            ("api_key", None),
            ("max_concurrent", None),
            ("rate_limit", None),
            ("max_retries", 3),
            ("retry_delay", 1.0),
            ("timeout", 300.0),
            ("input_cost_per_1m", 0.0),
            ("output_cost_per_1m", 0.0),
        ],
    )
    def test_defaults(self, field: str, default: object) -> None:
        """EndpointConfig has correct default values."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert getattr(config, field) == default

    @pytest.mark.parametrize("provider_api", ["openai", "anthropic", "vllm"])
    def test_accepts_all_providers(self, provider_api: str) -> None:
        """EndpointConfig accepts all supported provider API types."""
        config = EndpointConfig(provider_api=provider_api, model="test")  # type: ignore[arg-type]
        assert config.provider_api == provider_api

    def test_equality(self) -> None:
        """EndpointConfigs with same values are equal."""
        config1 = EndpointConfig(provider_api="openai", model="gpt-4o")
        config2 = EndpointConfig(provider_api="openai", model="gpt-4o")
        config3 = EndpointConfig(provider_api="openai", model="gpt-4o-mini")

        assert config1 == config2
        assert config1 != config3


class TestEndpointConfigApiKey:
    """Tests for EndpointConfig.get_api_key() method."""

    def test_returns_none_when_unset(self) -> None:
        """get_api_key returns None when api_key is not set."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert config.get_api_key() is None

    def test_returns_literal_value(self) -> None:
        """get_api_key returns literal value when not an env var."""
        config = EndpointConfig(
            provider_api="openai", model="gpt-4o", api_key="sk-my-literal-key"
        )
        assert config.get_api_key() == "sk-my-literal-key"

    def test_resolves_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_api_key reads from environment when api_key matches env var name."""
        monkeypatch.setenv("MY_CUSTOM_API_KEY", "secret-from-env")
        config = EndpointConfig(
            provider_api="openai", model="gpt-4o", api_key="MY_CUSTOM_API_KEY"
        )
        assert config.get_api_key() == "secret-from-env"

    def test_literal_when_env_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """api_key is returned as literal when no matching env var exists."""
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        config = EndpointConfig(
            provider_api="openai", model="gpt-4o", api_key="NONEXISTENT_VAR"
        )
        assert config.get_api_key() == "NONEXISTENT_VAR"

    def test_empty_env_var_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty string env var is still returned."""
        monkeypatch.setenv("EMPTY_KEY", "")
        config = EndpointConfig(
            provider_api="openai", model="gpt-4o", api_key="EMPTY_KEY"
        )
        assert config.get_api_key() == ""


class TestEndpointConfigHashKey:
    """Tests for EndpointConfig.hash_key property."""

    def test_is_valid_sha256_hex(self) -> None:
        """hash_key returns a valid SHA256 hex string."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert len(config.hash_key) == 64
        assert all(c in "0123456789abcdef" for c in config.hash_key)

    def test_same_endpoint_same_hash(self) -> None:
        """Configs with same base_url and api_key have same hash_key."""
        config1 = EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            base_url="https://api.openai.com",
            api_key="sk-test-key",
        )
        config2 = EndpointConfig(
            provider_api="openai",
            model="gpt-4o-mini",  # Different model
            base_url="https://api.openai.com",
            api_key="sk-test-key",
        )
        assert config1.hash_key == config2.hash_key

    def test_different_base_url_different_hash(self) -> None:
        """Configs with different base_url have different hash_key."""
        config1 = EndpointConfig(
            provider_api="vllm", model="mistral-7b", base_url="http://vllm-1:8000"
        )
        config2 = EndpointConfig(
            provider_api="vllm", model="mistral-7b", base_url="http://vllm-2:8000"
        )
        assert config1.hash_key != config2.hash_key

    def test_resolves_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """hash_key uses resolved api_key from environment."""
        monkeypatch.setenv("MY_API_KEY", "resolved-secret")
        config1 = EndpointConfig(
            provider_api="openai", model="gpt-4o", api_key="MY_API_KEY"
        )
        config2 = EndpointConfig(
            provider_api="openai", model="gpt-4o", api_key="resolved-secret"
        )
        assert config1.hash_key == config2.hash_key


class TestPresetConfigs:
    """Tests for preset endpoint configurations."""

    @pytest.mark.parametrize(
        "preset_class,provider_api,api_key_env,base_url",
        [
            (OpenAIEndpointConfig, "openai", "OPENAI_API_KEY", None),
            (AnthropicEndpointConfig, "anthropic", "ANTHROPIC_API_KEY", None),
            (
                NvidiaBuildEndpointConfig,
                "openai",
                "NVIDIA_API_KEY",
                "https://integrate.api.nvidia.com/v1",
            ),
        ],
    )
    def test_preset_config(
        self,
        preset_class: type,
        provider_api: str,
        api_key_env: str,
        base_url: str | None,
    ) -> None:
        """Preset configs have correct provider, api_key, and base_url."""
        config = preset_class(model="test-model")
        assert isinstance(config, EndpointConfig)
        assert config.provider_api == provider_api
        assert config.api_key == api_key_env
        assert config.base_url == base_url

    def test_preset_passes_through_kwargs(self) -> None:
        """Preset configs pass through additional kwargs."""
        config = OpenAIEndpointConfig(model="gpt-4o", max_concurrent=50, timeout=600.0)
        assert config.max_concurrent == 50
        assert config.timeout == 600.0

    def test_preset_resolves_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Preset configs resolve api_key from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-from-env")
        config = OpenAIEndpointConfig(model="gpt-4o")
        assert config.get_api_key() == "sk-test-key-from-env"

    def test_nvidia_hash_differs_from_openai(self) -> None:
        """NVIDIA and OpenAI configs have different hash due to base_url."""
        nvidia = NvidiaBuildEndpointConfig(
            model="meta/llama-3.1-405b-instruct", api_key="test-key"
        )
        openai = OpenAIEndpointConfig(model="gpt-4o", api_key="test-key")
        assert nvidia.hash_key != openai.hash_key


class TestResourceConfig:
    """Tests for ResourceConfig container."""

    def test_creation_and_access(self) -> None:
        """ResourceConfig stores and provides access to endpoints."""
        endpoint = EndpointConfig(provider_api="openai", model="gpt-4o-mini")
        config = ResourceConfig(endpoints={"fast": endpoint})

        assert len(config) == 1
        assert "fast" in config
        assert config["fast"] is endpoint
        assert config.get("fast") is endpoint
        assert config.get("nonexistent") is None

    def test_multiple_endpoints(self) -> None:
        """ResourceConfig can hold multiple endpoints."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini"),
                "smart": EndpointConfig(provider_api="openai", model="gpt-4o"),
                "local": EndpointConfig(
                    provider_api="vllm",
                    model="mistral-7b",
                    base_url="http://localhost:8000",
                ),
            }
        )

        assert len(config) == 3
        assert set(config.keys()) == {"fast", "smart", "local"}

    def test_dict_like_interface(self) -> None:
        """ResourceConfig provides dict-like iteration."""
        endpoint1 = EndpointConfig(provider_api="openai", model="gpt-4o-mini")
        endpoint2 = EndpointConfig(provider_api="openai", model="gpt-4o")
        config = ResourceConfig(endpoints={"fast": endpoint1, "smart": endpoint2})

        # keys, values, items
        assert set(config.keys()) == {"fast", "smart"}
        assert endpoint1 in config.values()
        assert endpoint2 in config.values()
        assert dict(config.items()) == {"fast": endpoint1, "smart": endpoint2}

        # iteration
        assert set(config) == {"fast", "smart"}

    def test_keyerror_for_missing(self) -> None:
        """__getitem__ raises KeyError for unknown alias."""
        config = ResourceConfig(
            endpoints={"fast": EndpointConfig(provider_api="openai", model="gpt-4o")}
        )
        with pytest.raises(KeyError):
            _ = config["nonexistent"]

    def test_contains_handles_non_string(self) -> None:
        """__contains__ handles non-string types gracefully."""
        config = ResourceConfig(
            endpoints={"fast": EndpointConfig(provider_api="openai", model="gpt-4o")}
        )
        assert 123 not in config
        assert None not in config

    def test_equality(self) -> None:
        """ResourceConfigs with same values are equal."""
        config1 = ResourceConfig(
            endpoints={"fast": EndpointConfig(provider_api="openai", model="gpt-4o")}
        )
        config2 = ResourceConfig(
            endpoints={"fast": EndpointConfig(provider_api="openai", model="gpt-4o")}
        )
        config3 = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
            }
        )

        assert config1 == config2
        assert config1 != config3

    def test_get_with_default(self) -> None:
        """get() returns specified default for nonexistent alias."""
        config = ResourceConfig(endpoints={})
        default = EndpointConfig(provider_api="vllm", model="fallback")

        result = config.get("missing", default)
        assert result is default


class TestResourceConfigUsagePatterns:
    """Tests for typical ResourceConfig usage patterns from design doc."""

    def test_dev_config_pattern(self) -> None:
        """Development configuration pattern works as documented."""
        dev_resources = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai", model="gpt-4o-mini", max_concurrent=5
                ),
                "smart": EndpointConfig(
                    provider_api="openai", model="gpt-4o", max_concurrent=2
                ),
            }
        )

        assert dev_resources["fast"].model == "gpt-4o-mini"
        assert dev_resources["smart"].model == "gpt-4o"
        assert dev_resources["fast"].max_concurrent == 5

    def test_hybrid_config_pattern(self) -> None:
        """Hybrid cloud/local configuration pattern works."""
        hybrid_resources = ResourceConfig(
            endpoints={
                "expensive": AnthropicEndpointConfig(
                    model="claude-sonnet-4-20250514",
                    max_concurrent=5,
                    input_cost_per_1m=3.0,
                    output_cost_per_1m=15.0,
                ),
                "fast": EndpointConfig(
                    provider_api="vllm",
                    model="llama3.2",
                    base_url="http://localhost:11434",
                    max_concurrent=4,
                ),
            }
        )

        assert hybrid_resources["expensive"].provider_api == "anthropic"
        assert hybrid_resources["fast"].base_url == "http://localhost:11434"

    def test_with_preset_configs(self) -> None:
        """ResourceConfig works with preset endpoint configs."""
        config = ResourceConfig(
            endpoints={
                "openai": OpenAIEndpointConfig(model="gpt-4o"),
                "anthropic": AnthropicEndpointConfig(model="claude-sonnet-4-20250514"),
                "nvidia": NvidiaBuildEndpointConfig(
                    model="meta/llama-3.1-405b-instruct"
                ),
            }
        )

        assert config["openai"].provider_api == "openai"
        assert config["anthropic"].provider_api == "anthropic"
        assert config["nvidia"].provider_api == "openai"  # OpenAI-compatible
        assert config["nvidia"].base_url == "https://integrate.api.nvidia.com/v1"
