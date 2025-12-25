"""Unit tests for resource configuration."""

import pytest

from inf_engine.resources.config import (
    AnthropicEndpointConfig,
    EndpointConfig,
    NvidiaBuildEndpointConfig,
    OpenAIEndpointConfig,
    ResourceConfig,
)


class TestEndpointConfigCreation:
    """Tests for EndpointConfig instantiation."""

    def test_endpoint_config_minimal(self) -> None:
        """EndpointConfig can be created with only required fields."""
        config = EndpointConfig(
            provider_api="openai",
            model="gpt-4o-mini",
        )

        assert config.provider_api == "openai"
        assert config.model == "gpt-4o-mini"

    def test_endpoint_config_all_fields(self) -> None:
        """EndpointConfig can be created with all fields specified."""
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
        "provider_api",
        ["openai", "anthropic", "vllm"],
    )
    def test_endpoint_config_all_providers(self, provider_api: str) -> None:
        """EndpointConfig accepts all supported provider API types."""
        config = EndpointConfig(
            provider_api=provider_api,  # type: ignore[arg-type]
            model="test-model",
        )

        assert config.provider_api == provider_api


class TestEndpointConfigDefaults:
    """Tests for EndpointConfig default values."""

    def test_base_url_defaults_none(self) -> None:
        """base_url defaults to None."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert config.base_url is None

    def test_api_key_defaults_none(self) -> None:
        """api_key defaults to None."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert config.api_key is None

    def test_max_concurrent_defaults_none(self) -> None:
        """max_concurrent defaults to None for adaptive backpressure."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert config.max_concurrent is None

    def test_rate_limit_defaults_none(self) -> None:
        """rate_limit defaults to None (no rate limiting)."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert config.rate_limit is None

    def test_max_retries_defaults_3(self) -> None:
        """max_retries defaults to 3."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert config.max_retries == 3

    def test_retry_delay_defaults_1(self) -> None:
        """retry_delay defaults to 1.0 second."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert config.retry_delay == 1.0

    def test_timeout_defaults_300(self) -> None:
        """timeout defaults to 300 seconds (5 minutes)."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert config.timeout == 300.0

    def test_input_cost_defaults_zero(self) -> None:
        """input_cost_per_1m defaults to 0.0."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert config.input_cost_per_1m == 0.0

    def test_output_cost_defaults_zero(self) -> None:
        """output_cost_per_1m defaults to 0.0."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert config.output_cost_per_1m == 0.0


class TestEndpointConfigUsagePatterns:
    """Tests for typical EndpointConfig usage patterns."""

    def test_openai_config(self) -> None:
        """Typical OpenAI configuration."""
        config = EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            max_concurrent=10,
            input_cost_per_1m=2.50,
            output_cost_per_1m=10.0,
        )

        assert config.provider_api == "openai"
        assert config.base_url is None  # Uses default OpenAI URL

    def test_self_hosted_vllm_config(self) -> None:
        """Self-hosted vLLM configuration."""
        config = EndpointConfig(
            provider_api="vllm",
            model="mistral-7b",
            base_url="http://vllm.internal:8000",
            max_concurrent=100,
            rate_limit=50.0,
        )

        assert config.provider_api == "vllm"
        assert config.base_url == "http://vllm.internal:8000"
        assert config.api_key is None  # Self-hosted typically doesn't need auth

    def test_anthropic_config(self) -> None:
        """Anthropic Claude configuration."""
        config = EndpointConfig(
            provider_api="anthropic",
            model="claude-3-opus",
            max_concurrent=5,
            input_cost_per_1m=15.0,
            output_cost_per_1m=75.0,
        )

        assert config.provider_api == "anthropic"
        assert config.input_cost_per_1m == 15.0
        assert config.output_cost_per_1m == 75.0


class TestEndpointConfigEquality:
    """Tests for EndpointConfig equality comparison."""

    def test_equal_configs(self) -> None:
        """Two configs with same values are equal."""
        config1 = EndpointConfig(provider_api="openai", model="gpt-4o")
        config2 = EndpointConfig(provider_api="openai", model="gpt-4o")

        assert config1 == config2

    def test_unequal_configs(self) -> None:
        """Two configs with different values are not equal."""
        config1 = EndpointConfig(provider_api="openai", model="gpt-4o")
        config2 = EndpointConfig(provider_api="openai", model="gpt-4o-mini")

        assert config1 != config2


class TestEndpointConfigGetApiKey:
    """Tests for EndpointConfig.get_api_key() method."""

    def test_get_api_key_none(self) -> None:
        """get_api_key returns None when api_key is not set."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert config.get_api_key() is None

    def test_get_api_key_literal_value(self) -> None:
        """get_api_key returns literal value when not an env var."""
        config = EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            api_key="sk-my-literal-key",
        )
        assert config.get_api_key() == "sk-my-literal-key"

    def test_get_api_key_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_api_key reads from environment when api_key matches env var name."""
        monkeypatch.setenv("MY_CUSTOM_API_KEY", "secret-from-env")

        config = EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            api_key="MY_CUSTOM_API_KEY",
        )

        assert config.get_api_key() == "secret-from-env"

    def test_get_api_key_env_var_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If api_key matches an env var name, env var value is used."""
        # Set an env var that happens to match a potential literal key
        monkeypatch.setenv("OPENAI_API_KEY", "env-value")

        config = EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            api_key="OPENAI_API_KEY",
        )

        assert config.get_api_key() == "env-value"

    def test_get_api_key_literal_when_env_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """api_key is returned as literal when no matching env var exists."""
        # Ensure the env var doesn't exist
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)

        config = EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            api_key="NONEXISTENT_VAR",
        )

        # Since NONEXISTENT_VAR is not in environment, treat as literal
        assert config.get_api_key() == "NONEXISTENT_VAR"

    def test_get_api_key_empty_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty string env var is still returned (not treated as missing)."""
        monkeypatch.setenv("EMPTY_KEY", "")

        config = EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            api_key="EMPTY_KEY",
        )

        assert config.get_api_key() == ""


class TestEndpointConfigHashKey:
    """Tests for EndpointConfig.hash_key property."""

    def test_hash_key_is_string(self) -> None:
        """hash_key returns a string."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        assert isinstance(config.hash_key, str)

    def test_hash_key_is_hex(self) -> None:
        """hash_key returns a valid hex string."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")
        # SHA256 produces 64 hex characters
        assert len(config.hash_key) == 64
        assert all(c in "0123456789abcdef" for c in config.hash_key)

    def test_hash_key_same_endpoint_same_hash(self) -> None:
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
            base_url="https://api.openai.com",  # Same base_url
            api_key="sk-test-key",  # Same api_key
        )

        assert config1.hash_key == config2.hash_key

    def test_hash_key_different_base_url(self) -> None:
        """Configs with different base_url have different hash_key."""
        config1 = EndpointConfig(
            provider_api="vllm",
            model="mistral-7b",
            base_url="http://vllm-1.internal:8000",
        )
        config2 = EndpointConfig(
            provider_api="vllm",
            model="mistral-7b",
            base_url="http://vllm-2.internal:8000",
        )

        assert config1.hash_key != config2.hash_key

    def test_hash_key_different_api_key(self) -> None:
        """Configs with different api_key have different hash_key."""
        config1 = EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            api_key="sk-key-1",
        )
        config2 = EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            api_key="sk-key-2",
        )

        assert config1.hash_key != config2.hash_key

    def test_hash_key_none_values(self) -> None:
        """hash_key works when base_url and api_key are None."""
        config = EndpointConfig(provider_api="openai", model="gpt-4o")

        # Should not raise, should return consistent hash
        hash1 = config.hash_key
        hash2 = config.hash_key
        assert hash1 == hash2

    def test_hash_key_resolves_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """hash_key uses resolved api_key from environment."""
        monkeypatch.setenv("MY_API_KEY", "resolved-secret")

        config1 = EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            api_key="MY_API_KEY",  # Env var name
        )
        config2 = EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            api_key="resolved-secret",  # Literal value (same as env var value)
        )

        # Both should have same hash since they resolve to same api_key
        assert config1.hash_key == config2.hash_key

    def test_hash_key_deterministic(self) -> None:
        """hash_key is deterministic across multiple calls."""
        config = EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            base_url="https://api.example.com",
            api_key="test-key",
        )

        hashes = [config.hash_key for _ in range(10)]
        assert all(h == hashes[0] for h in hashes)


class TestOpenAIEndpointConfig:
    """Tests for OpenAIEndpointConfig preset."""

    def test_minimal_creation(self) -> None:
        """OpenAIEndpointConfig can be created with just model."""
        config = OpenAIEndpointConfig(model="gpt-4o-mini")

        assert config.model == "gpt-4o-mini"
        assert config.provider_api == "openai"
        assert config.api_key == "OPENAI_API_KEY"
        assert config.base_url is None

    def test_inherits_from_endpoint_config(self) -> None:
        """OpenAIEndpointConfig is an EndpointConfig instance."""
        config = OpenAIEndpointConfig(model="gpt-4o")
        assert isinstance(config, EndpointConfig)

    def test_custom_api_key(self) -> None:
        """OpenAIEndpointConfig accepts custom api_key."""
        config = OpenAIEndpointConfig(model="gpt-4o", api_key="MY_CUSTOM_KEY")
        assert config.api_key == "MY_CUSTOM_KEY"

    def test_additional_kwargs(self) -> None:
        """OpenAIEndpointConfig passes through additional kwargs."""
        config = OpenAIEndpointConfig(
            model="gpt-4o",
            max_concurrent=50,
            timeout=600.0,
        )

        assert config.max_concurrent == 50
        assert config.timeout == 600.0

    def test_get_api_key_resolves_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OpenAIEndpointConfig resolves OPENAI_API_KEY from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-from-env")

        config = OpenAIEndpointConfig(model="gpt-4o")
        assert config.get_api_key() == "sk-test-key-from-env"

    def test_hash_key_works(self) -> None:
        """OpenAIEndpointConfig has working hash_key property."""
        config = OpenAIEndpointConfig(model="gpt-4o")
        assert isinstance(config.hash_key, str)
        assert len(config.hash_key) == 64


class TestAnthropicEndpointConfig:
    """Tests for AnthropicEndpointConfig preset."""

    def test_minimal_creation(self) -> None:
        """AnthropicEndpointConfig can be created with just model."""
        config = AnthropicEndpointConfig(model="claude-sonnet-4-20250514")

        assert config.model == "claude-sonnet-4-20250514"
        assert config.provider_api == "anthropic"
        assert config.api_key == "ANTHROPIC_API_KEY"
        assert config.base_url is None

    def test_inherits_from_endpoint_config(self) -> None:
        """AnthropicEndpointConfig is an EndpointConfig instance."""
        config = AnthropicEndpointConfig(model="claude-sonnet-4-20250514")
        assert isinstance(config, EndpointConfig)

    def test_custom_api_key(self) -> None:
        """AnthropicEndpointConfig accepts custom api_key."""
        config = AnthropicEndpointConfig(
            model="claude-sonnet-4-20250514",
            api_key="MY_ANTHROPIC_KEY",
        )
        assert config.api_key == "MY_ANTHROPIC_KEY"

    def test_additional_kwargs(self) -> None:
        """AnthropicEndpointConfig passes through additional kwargs."""
        config = AnthropicEndpointConfig(
            model="claude-sonnet-4-20250514",
            max_concurrent=20,
            input_cost_per_1m=3.0,
        )

        assert config.max_concurrent == 20
        assert config.input_cost_per_1m == 3.0

    def test_get_api_key_resolves_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AnthropicEndpointConfig resolves ANTHROPIC_API_KEY from environment."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

        config = AnthropicEndpointConfig(model="claude-sonnet-4-20250514")
        assert config.get_api_key() == "sk-ant-test-key"


class TestNvidiaBuildEndpointConfig:
    """Tests for NvidiaBuildEndpointConfig preset."""

    def test_minimal_creation(self) -> None:
        """NvidiaBuildEndpointConfig can be created with just model."""
        config = NvidiaBuildEndpointConfig(model="meta/llama-3.1-405b-instruct")

        assert config.model == "meta/llama-3.1-405b-instruct"
        assert config.provider_api == "openai"  # Uses OpenAI-compatible API
        assert config.api_key == "NVIDIA_API_KEY"
        assert config.base_url == "https://integrate.api.nvidia.com/v1"

    def test_inherits_from_endpoint_config(self) -> None:
        """NvidiaBuildEndpointConfig is an EndpointConfig instance."""
        config = NvidiaBuildEndpointConfig(model="meta/llama-3.1-405b-instruct")
        assert isinstance(config, EndpointConfig)

    def test_custom_api_key(self) -> None:
        """NvidiaBuildEndpointConfig accepts custom api_key."""
        config = NvidiaBuildEndpointConfig(
            model="meta/llama-3.1-405b-instruct",
            api_key="MY_NVIDIA_KEY",
        )
        assert config.api_key == "MY_NVIDIA_KEY"

    def test_base_url_is_set(self) -> None:
        """NvidiaBuildEndpointConfig has correct base_url."""
        config = NvidiaBuildEndpointConfig(model="nvidia/nemotron-4-340b-instruct")
        assert config.base_url == "https://integrate.api.nvidia.com/v1"

    def test_additional_kwargs(self) -> None:
        """NvidiaBuildEndpointConfig passes through additional kwargs."""
        config = NvidiaBuildEndpointConfig(
            model="meta/llama-3.1-405b-instruct",
            max_concurrent=100,
            rate_limit=50.0,
        )

        assert config.max_concurrent == 100
        assert config.rate_limit == 50.0

    def test_get_api_key_resolves_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NvidiaBuildEndpointConfig resolves NVIDIA_API_KEY from environment."""
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")

        config = NvidiaBuildEndpointConfig(model="meta/llama-3.1-405b-instruct")
        assert config.get_api_key() == "nvapi-test-key"

    def test_hash_key_includes_base_url(self) -> None:
        """NvidiaBuildEndpointConfig hash_key includes the NVIDIA base_url."""
        config1 = NvidiaBuildEndpointConfig(
            model="meta/llama-3.1-405b-instruct",
            api_key="test-key",
        )
        config2 = OpenAIEndpointConfig(
            model="gpt-4o",
            api_key="test-key",  # Same api_key but different base_url
        )

        # Different hash because different base_url
        assert config1.hash_key != config2.hash_key


class TestResourceConfigCreation:
    """Tests for ResourceConfig instantiation."""

    def test_minimal_creation(self) -> None:
        """ResourceConfig can be created with only endpoints."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini"),
            }
        )

        assert "fast" in config.endpoints
        assert config.endpoints["fast"].model == "gpt-4o-mini"

    def test_empty_endpoints(self) -> None:
        """ResourceConfig can be created with empty endpoints dict."""
        config = ResourceConfig(endpoints={})

        assert len(config.endpoints) == 0

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

        assert len(config.endpoints) == 3
        assert "fast" in config.endpoints
        assert "smart" in config.endpoints
        assert "local" in config.endpoints


class TestResourceConfigGetItem:
    """Tests for ResourceConfig.__getitem__()."""

    def test_getitem_returns_endpoint(self) -> None:
        """__getitem__ returns the endpoint for a valid alias."""
        endpoint = EndpointConfig(provider_api="openai", model="gpt-4o")
        config = ResourceConfig(endpoints={"smart": endpoint})

        assert config["smart"] is endpoint

    def test_getitem_raises_keyerror(self) -> None:
        """__getitem__ raises KeyError for unknown alias."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
            }
        )

        with pytest.raises(KeyError):
            _ = config["nonexistent"]

    def test_getitem_multiple_aliases(self) -> None:
        """__getitem__ works with multiple aliases."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini"),
                "smart": EndpointConfig(provider_api="openai", model="gpt-4o"),
            }
        )

        assert config["fast"].model == "gpt-4o-mini"
        assert config["smart"].model == "gpt-4o"


class TestResourceConfigContains:
    """Tests for ResourceConfig.__contains__()."""

    def test_contains_existing_alias(self) -> None:
        """__contains__ returns True for existing alias."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
            }
        )

        assert "fast" in config

    def test_contains_nonexistent_alias(self) -> None:
        """__contains__ returns False for nonexistent alias."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
            }
        )

        assert "slow" not in config

    def test_contains_empty_config(self) -> None:
        """__contains__ works with empty endpoints."""
        config = ResourceConfig(endpoints={})

        assert "any" not in config

    def test_contains_non_string_types(self) -> None:
        """__contains__ handles non-string types gracefully."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
            }
        )

        # Non-string types should return False, not raise
        assert 123 not in config
        assert None not in config
        assert 3.14 not in config
        assert [] not in config


class TestResourceConfigIter:
    """Tests for ResourceConfig.__iter__()."""

    def test_iter_returns_aliases(self) -> None:
        """__iter__ yields alias names."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini"),
                "smart": EndpointConfig(provider_api="openai", model="gpt-4o"),
            }
        )

        aliases = list(config)
        assert set(aliases) == {"fast", "smart"}

    def test_iter_empty_config(self) -> None:
        """__iter__ works with empty endpoints."""
        config = ResourceConfig(endpoints={})

        assert list(config) == []

    def test_iter_for_loop(self) -> None:
        """ResourceConfig can be used in a for loop."""
        config = ResourceConfig(
            endpoints={
                "a": EndpointConfig(provider_api="openai", model="m1"),
                "b": EndpointConfig(provider_api="openai", model="m2"),
            }
        )

        aliases = []
        for alias in config:
            aliases.append(alias)

        assert set(aliases) == {"a", "b"}


class TestResourceConfigLen:
    """Tests for ResourceConfig.__len__()."""

    def test_len_single_endpoint(self) -> None:
        """__len__ returns 1 for single endpoint."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
            }
        )

        assert len(config) == 1

    def test_len_multiple_endpoints(self) -> None:
        """__len__ returns correct count for multiple endpoints."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini"),
                "smart": EndpointConfig(provider_api="openai", model="gpt-4o"),
                "local": EndpointConfig(provider_api="vllm", model="mistral-7b"),
            }
        )

        assert len(config) == 3

    def test_len_empty_config(self) -> None:
        """__len__ returns 0 for empty endpoints."""
        config = ResourceConfig(endpoints={})

        assert len(config) == 0


class TestResourceConfigKeys:
    """Tests for ResourceConfig.keys()."""

    def test_keys_returns_alias_names(self) -> None:
        """keys() returns all alias names."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini"),
                "smart": EndpointConfig(provider_api="openai", model="gpt-4o"),
            }
        )

        assert set(config.keys()) == {"fast", "smart"}

    def test_keys_empty_config(self) -> None:
        """keys() returns empty for empty endpoints."""
        config = ResourceConfig(endpoints={})

        assert list(config.keys()) == []


class TestResourceConfigValues:
    """Tests for ResourceConfig.values()."""

    def test_values_returns_endpoints(self) -> None:
        """values() returns all EndpointConfig instances."""
        endpoint1 = EndpointConfig(provider_api="openai", model="gpt-4o-mini")
        endpoint2 = EndpointConfig(provider_api="openai", model="gpt-4o")
        config = ResourceConfig(endpoints={"fast": endpoint1, "smart": endpoint2})

        values = list(config.values())
        assert len(values) == 2
        assert endpoint1 in values
        assert endpoint2 in values

    def test_values_empty_config(self) -> None:
        """values() returns empty for empty endpoints."""
        config = ResourceConfig(endpoints={})

        assert list(config.values()) == []


class TestResourceConfigItems:
    """Tests for ResourceConfig.items()."""

    def test_items_returns_pairs(self) -> None:
        """items() returns (alias, endpoint) pairs."""
        endpoint1 = EndpointConfig(provider_api="openai", model="gpt-4o-mini")
        endpoint2 = EndpointConfig(provider_api="openai", model="gpt-4o")
        config = ResourceConfig(endpoints={"fast": endpoint1, "smart": endpoint2})

        items = dict(config.items())
        assert items["fast"] is endpoint1
        assert items["smart"] is endpoint2

    def test_items_empty_config(self) -> None:
        """items() returns empty for empty endpoints."""
        config = ResourceConfig(endpoints={})

        assert list(config.items()) == []


class TestResourceConfigGet:
    """Tests for ResourceConfig.get()."""

    def test_get_existing_alias(self) -> None:
        """get() returns endpoint for existing alias."""
        endpoint = EndpointConfig(provider_api="openai", model="gpt-4o")
        config = ResourceConfig(endpoints={"smart": endpoint})

        assert config.get("smart") is endpoint

    def test_get_nonexistent_returns_none(self) -> None:
        """get() returns None for nonexistent alias by default."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
            }
        )

        assert config.get("nonexistent") is None

    def test_get_with_default(self) -> None:
        """get() returns specified default for nonexistent alias."""
        config = ResourceConfig(endpoints={})
        default = EndpointConfig(provider_api="vllm", model="fallback")

        result = config.get("missing", default)
        assert result is default

    def test_get_existing_ignores_default(self) -> None:
        """get() returns endpoint even when default is provided."""
        endpoint = EndpointConfig(provider_api="openai", model="gpt-4o")
        default = EndpointConfig(provider_api="vllm", model="fallback")
        config = ResourceConfig(endpoints={"smart": endpoint})

        result = config.get("smart", default)
        assert result is endpoint


class TestResourceConfigEquality:
    """Tests for ResourceConfig equality comparison."""

    def test_equal_configs(self) -> None:
        """Two configs with same values are equal."""
        config1 = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
            },
        )
        config2 = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
            },
        )

        assert config1 == config2

    def test_unequal_endpoints(self) -> None:
        """Configs with different endpoints are not equal."""
        config1 = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
            }
        )
        config2 = ResourceConfig(
            endpoints={"fast": EndpointConfig(provider_api="openai", model="gpt-4o")}
        )

        assert config1 != config2


class TestResourceConfigUsagePatterns:
    """Tests for typical ResourceConfig usage patterns from design doc."""

    def test_dev_config_pattern(self) -> None:
        """Development configuration pattern works as documented."""
        dev_resources = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=5,
                ),
                "smart": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o",
                    max_concurrent=2,
                ),
            },
        )

        assert dev_resources["fast"].model == "gpt-4o-mini"
        assert dev_resources["smart"].model == "gpt-4o"
        assert dev_resources["fast"].max_concurrent == 5
        assert len(dev_resources) == 2

    def test_prod_config_pattern(self) -> None:
        """Production configuration pattern works as documented."""
        prod_resources = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="vllm",
                    model="mistral-7b",
                    base_url="http://vllm-fast.internal:8000",
                    max_concurrent=50,
                    rate_limit=100.0,
                ),
                "smart": EndpointConfig(
                    provider_api="vllm",
                    model="llama-70b",
                    base_url="http://vllm-smart.internal:8000",
                    max_concurrent=20,
                    rate_limit=30.0,
                ),
            },
        )

        assert prod_resources["fast"].base_url == "http://vllm-fast.internal:8000"
        assert prod_resources["smart"].rate_limit == 30.0
        assert prod_resources["fast"].max_concurrent == 50
        assert prod_resources["smart"].max_concurrent == 20

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
            },
        )

        assert hybrid_resources["expensive"].provider_api == "anthropic"
        assert hybrid_resources["fast"].base_url == "http://localhost:11434"

    def test_iteration_pattern(self) -> None:
        """Common iteration pattern for accessing all endpoints."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini"),
                "smart": EndpointConfig(provider_api="openai", model="gpt-4o"),
            }
        )

        # Pattern: iterate and access endpoints
        models = []
        for alias in config:
            models.append(config[alias].model)

        assert set(models) == {"gpt-4o-mini", "gpt-4o"}

    def test_items_iteration_pattern(self) -> None:
        """Common items() iteration pattern."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini"),
                "smart": EndpointConfig(provider_api="openai", model="gpt-4o"),
            }
        )

        # Pattern: iterate with items()
        alias_model_pairs = {alias: ep.model for alias, ep in config.items()}

        assert alias_model_pairs == {"fast": "gpt-4o-mini", "smart": "gpt-4o"}

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
        assert (
            config["nvidia"].provider_api == "openai"
        )  # NVIDIA uses OpenAI-compatible
        assert config["nvidia"].base_url == "https://integrate.api.nvidia.com/v1"
