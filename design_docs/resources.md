# Resource Management

Resource management in inf-engine separates module definitions from infrastructure configuration. Modules declare what they need via aliases; the resource layer binds these to actual endpoints.

## Design Goals

1. **Separation of Concerns**: Modules don't know about specific endpoints
2. **Flexibility**: Same module can use different backends (OpenAI, vLLM, local)
3. **Efficiency**: Connection pooling, rate limiting, load balancing
4. **Observability**: Track utilization, latency, costs

## Resource Configuration

### Configuration Structure

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class EndpointConfig:
    """Configuration for a single LLM endpoint."""

    provider: Literal["openai", "anthropic", "vllm", "tgi", "ollama"]
    model: str
    base_url: str | None = None       # Custom endpoint URL
    api_key: str | None = None         # API key (or from env)

    # Concurrency limits
    max_concurrent: int = 10           # Max parallel requests
    rate_limit: float | None = None    # Requests per second

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0

    # Timeout
    timeout: float = 300.0             # Seconds

    # Cost tracking
    input_cost_per_1m: float = 0.0     # $ per 1M input tokens
    output_cost_per_1m: float = 0.0    # $ per 1M output tokens


@dataclass
class ResourceConfig:
    """Complete resource configuration."""

    endpoints: dict[str, EndpointConfig]

    # Global limits
    max_total_concurrent: int = 100
    max_inflight_graphs: int = 10

    # Memory management
    result_cache_size: int = 1000      # Max cached results
    checkpoint_buffer_size: int = 10    # Completions before flush

    def __getitem__(self, alias: str) -> EndpointConfig:
        return self.endpoints[alias]

    def __contains__(self, alias: str) -> bool:
        return alias in self.endpoints
```

### Configuration Examples

```python
# Development configuration
dev_resources = ResourceConfig(
    endpoints={
        "fast": EndpointConfig(
            provider="openai",
            model="gpt-4o-mini",
            max_concurrent=5,
        ),
        "smart": EndpointConfig(
            provider="openai",
            model="gpt-4o",
            max_concurrent=2,
        ),
    },
    max_total_concurrent=10,
)

# Production with self-hosted models
prod_resources = ResourceConfig(
    endpoints={
        "fast": EndpointConfig(
            provider="vllm",
            model="mistral-7b",
            base_url="http://vllm-fast.internal:8000",
            max_concurrent=50,
            rate_limit=100.0,
        ),
        "smart": EndpointConfig(
            provider="vllm",
            model="llama-70b",
            base_url="http://vllm-smart.internal:8000",
            max_concurrent=20,
            rate_limit=30.0,
        ),
        "embedding": EndpointConfig(
            provider="tgi",
            model="bge-large",
            base_url="http://tgi-embed.internal:8080",
            max_concurrent=100,
        ),
    },
    max_total_concurrent=200,
    max_inflight_graphs=50,
)

# Hybrid cloud/local
hybrid_resources = ResourceConfig(
    endpoints={
        "expensive": EndpointConfig(
            provider="anthropic",
            model="claude-3-opus",
            max_concurrent=5,
            input_cost_per_1m=15.0,
            output_cost_per_1m=75.0,
        ),
        "fast": EndpointConfig(
            provider="ollama",
            model="llama3.2",
            base_url="http://localhost:11434",
            max_concurrent=4,
        ),
    },
)
```

### Loading from Files

```python
# config/resources.yaml
endpoints:
  fast:
    provider: openai
    model: gpt-4o-mini
    max_concurrent: 20

  smart:
    provider: openai
    model: gpt-4o
    max_concurrent: 10

max_total_concurrent: 50
max_inflight_graphs: 20
```

```python
from inf_engine import ResourceConfig

resources = ResourceConfig.from_yaml("config/resources.yaml")
```

## Resource Manager

The `ResourceManager` handles all runtime resource coordination:

```python
class ResourceManager:
    """
    Manages LLM endpoints and coordinates resource access.
    """

    def __init__(self, config: ResourceConfig):
        self.config = config

        # Per-endpoint resources
        self.clients: dict[str, LLMClient] = {}
        self.semaphores: dict[str, asyncio.Semaphore] = {}
        self.rate_limiters: dict[str, RateLimiter] = {}

        # Global resources
        self._global_semaphore = asyncio.Semaphore(config.max_total_concurrent)

        # Metrics
        self.metrics = ResourceMetrics()

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize clients and limiters for each endpoint."""
        for alias, endpoint in self.config.endpoints.items():
            # Create client
            self.clients[alias] = self._create_client(endpoint)

            # Create semaphore
            self.semaphores[alias] = asyncio.Semaphore(endpoint.max_concurrent)

            # Create rate limiter
            if endpoint.rate_limit:
                self.rate_limiters[alias] = RateLimiter(
                    initial_rate=endpoint.rate_limit,
                    max_tokens=endpoint.max_concurrent,
                )

    def _create_client(self, endpoint: EndpointConfig) -> LLMClient:
        """Create the appropriate client for an endpoint."""
        match endpoint.provider:
            case "openai":
                return OpenAIClient(
                    model=endpoint.model,
                    base_url=endpoint.base_url,
                    api_key=endpoint.api_key,
                    timeout=endpoint.timeout,
                )
            case "anthropic":
                return AnthropicClient(
                    model=endpoint.model,
                    api_key=endpoint.api_key,
                    timeout=endpoint.timeout,
                )
            case "vllm" | "tgi":
                return OpenAICompatibleClient(
                    model=endpoint.model,
                    base_url=endpoint.base_url,
                    timeout=endpoint.timeout,
                )
            case "ollama":
                return OllamaClient(
                    model=endpoint.model,
                    base_url=endpoint.base_url or "http://localhost:11434",
                    timeout=endpoint.timeout,
                )
            case _:
                raise ValueError(f"Unknown provider: {endpoint.provider}")

    async def execute(
        self,
        alias: str,
        module: LLMInference,
        args: tuple,
        kwargs: dict,
    ) -> str:
        """
        Execute an LLM call with resource management.
        """
        if alias not in self.clients:
            raise ValueError(f"Unknown alias: {alias}")

        endpoint = self.config.endpoints[alias]
        client = self.clients[alias]

        # Acquire resources
        async with self._global_semaphore:
            async with self.semaphores[alias]:
                # Rate limiting
                if alias in self.rate_limiters:
                    await self.rate_limiters[alias].acquire()

                # Track metrics
                start_time = time.time()

                try:
                    # Build request
                    request = self._build_request(module, args, kwargs)

                    # Execute
                    response = await client.complete(request)

                    # Record success
                    duration = time.time() - start_time
                    self.metrics.record_success(alias, duration, response)

                    if alias in self.rate_limiters:
                        self.rate_limiters[alias].recover()

                    return response.content

                except RateLimitError as e:
                    # Propagate for requeue
                    self.metrics.record_rate_limit(alias)
                    raise

                except Exception as e:
                    duration = time.time() - start_time
                    self.metrics.record_error(alias, duration, e)
                    raise

    def _build_request(
        self,
        module: LLMInference,
        args: tuple,
        kwargs: dict,
    ) -> LLMRequest:
        """Build an LLM request from module and args."""
        # Get prompt from args
        prompt = args[0] if args else kwargs.get("prompt", "")

        # Get system prompt
        system_prompt = None
        if module.system_prompt:
            system_prompt = str(module.system_prompt)

        return LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=module.temperature,
            max_tokens=module.max_tokens,
            response_format=module.response_format,
        )

    def handle_rate_limit(
        self,
        alias: str | None,
        retry_after: float | None,
    ) -> None:
        """Handle a rate limit response."""
        if alias and alias in self.rate_limiters:
            self.rate_limiters[alias].backoff(retry_after)

    def get_stats(self) -> dict[str, Any]:
        """Get resource utilization statistics."""
        return {
            alias: {
                "available": self.semaphores[alias]._value,
                "max": self.config.endpoints[alias].max_concurrent,
                "metrics": self.metrics.get_alias_stats(alias),
            }
            for alias in self.clients
        }
```

## LLM Clients

Abstract client interface and implementations:

```python
from abc import ABC, abstractmethod

@dataclass
class LLMRequest:
    """A request to an LLM endpoint."""

    prompt: str
    system_prompt: str | None = None
    temperature: float = 1.0
    max_tokens: int | None = None
    response_format: type | None = None
    stop: list[str] | None = None


@dataclass
class LLMResponse:
    """A response from an LLM endpoint."""

    content: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    model: str


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a completion request."""
        pass


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
    ):
        import openai

        self.model = model
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop,
            )

            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                finish_reason=choice.finish_reason,
                model=response.model,
            )

        except openai.RateLimitError as e:
            # Extract retry-after if available
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    retry_after = float(retry_after)
            raise RateLimitError(retry_after=retry_after) from e


class OpenAICompatibleClient(OpenAIClient):
    """Client for OpenAI-compatible APIs (vLLM, TGI, etc.)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: float = 300.0,
    ):
        import openai

        self.model = model
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",  # Most local servers don't need auth
            timeout=timeout,
        )
```

## Metrics and Monitoring

Track resource utilization and costs:

```python
@dataclass
class EndpointMetrics:
    """Metrics for a single endpoint."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class ResourceMetrics:
    """Aggregated metrics across all endpoints."""

    def __init__(self):
        self._metrics: dict[str, EndpointMetrics] = defaultdict(EndpointMetrics)
        self._lock = threading.Lock()

    def record_success(
        self,
        alias: str,
        duration: float,
        response: LLMResponse,
    ) -> None:
        with self._lock:
            m = self._metrics[alias]
            m.total_requests += 1
            m.successful_requests += 1
            m.total_input_tokens += response.input_tokens
            m.total_output_tokens += response.output_tokens

            latency_ms = duration * 1000
            m.total_latency_ms += latency_ms
            m.min_latency_ms = min(m.min_latency_ms, latency_ms)
            m.max_latency_ms = max(m.max_latency_ms, latency_ms)

    def record_error(
        self,
        alias: str,
        duration: float,
        error: Exception,
    ) -> None:
        with self._lock:
            m = self._metrics[alias]
            m.total_requests += 1
            m.failed_requests += 1

    def record_rate_limit(self, alias: str) -> None:
        with self._lock:
            m = self._metrics[alias]
            m.total_requests += 1
            m.rate_limited_requests += 1

    def get_alias_stats(self, alias: str) -> dict[str, Any]:
        with self._lock:
            m = self._metrics[alias]
            return {
                "total_requests": m.total_requests,
                "success_rate": m.success_rate,
                "avg_latency_ms": m.avg_latency_ms,
                "total_tokens": m.total_input_tokens + m.total_output_tokens,
            }

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        return {alias: self.get_alias_stats(alias) for alias in self._metrics}

    def estimate_cost(self, config: ResourceConfig) -> float:
        """Estimate total cost based on token usage."""
        total = 0.0
        for alias, metrics in self._metrics.items():
            if alias in config.endpoints:
                endpoint = config.endpoints[alias]
                input_cost = (metrics.total_input_tokens / 1_000_000) * endpoint.input_cost_per_1m
                output_cost = (metrics.total_output_tokens / 1_000_000) * endpoint.output_cost_per_1m
                total += input_cost + output_cost
        return total
```

## Connection Pooling

For high-throughput scenarios:

```python
class ConnectionPool:
    """
    Manages a pool of connections to an endpoint.

    Useful for self-hosted models where connection reuse
    provides significant latency benefits.
    """

    def __init__(
        self,
        endpoint: EndpointConfig,
        pool_size: int = 10,
    ):
        self.endpoint = endpoint
        self.pool_size = pool_size
        self._pool: asyncio.Queue[LLMClient] = asyncio.Queue(maxsize=pool_size)
        self._initialized = False

    async def initialize(self) -> None:
        """Pre-create connections."""
        if self._initialized:
            return

        for _ in range(self.pool_size):
            client = self._create_client()
            await self._pool.put(client)

        self._initialized = True

    def _create_client(self) -> LLMClient:
        """Create a new client instance."""
        # Implementation depends on endpoint type
        ...

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[LLMClient]:
        """Acquire a connection from the pool."""
        client = await self._pool.get()
        try:
            yield client
        finally:
            await self._pool.put(client)
```

## Load Balancing

Distribute requests across multiple endpoints:

```python
class LoadBalancer:
    """
    Distributes requests across multiple endpoints for the same alias.
    """

    def __init__(
        self,
        endpoints: list[EndpointConfig],
        strategy: Literal["round_robin", "least_loaded", "random"] = "least_loaded",
    ):
        self.endpoints = endpoints
        self.strategy = strategy
        self._index = 0
        self._loads: dict[int, int] = {i: 0 for i in range(len(endpoints))}
        self._lock = asyncio.Lock()

    async def select(self) -> tuple[int, EndpointConfig]:
        """Select the next endpoint to use."""
        async with self._lock:
            match self.strategy:
                case "round_robin":
                    idx = self._index
                    self._index = (self._index + 1) % len(self.endpoints)
                    return idx, self.endpoints[idx]

                case "least_loaded":
                    idx = min(self._loads, key=self._loads.get)
                    self._loads[idx] += 1
                    return idx, self.endpoints[idx]

                case "random":
                    idx = random.randint(0, len(self.endpoints) - 1)
                    return idx, self.endpoints[idx]

    def release(self, idx: int) -> None:
        """Release a load slot."""
        self._loads[idx] = max(0, self._loads[idx] - 1)


# Configuration example
balanced_resources = ResourceConfig(
    endpoints={
        "fast": [
            EndpointConfig(provider="vllm", model="mistral-7b", base_url="http://vllm-1:8000"),
            EndpointConfig(provider="vllm", model="mistral-7b", base_url="http://vllm-2:8000"),
            EndpointConfig(provider="vllm", model="mistral-7b", base_url="http://vllm-3:8000"),
        ],
    },
    load_balance_strategy="least_loaded",
)
```

## Module-Resource Binding

How modules discover and use resources:

```python
class LLMInference(InferenceModule):
    """
    Atomic LLM operation.

    The alias connects this module to a configured endpoint.
    The actual endpoint is resolved at runtime by ResourceManager.
    """

    def __init__(self, alias: str, ...):
        self.alias = alias
        # Module is unaware of actual endpoint details

# During execution, ResourceManager binds alias to endpoint
async def execute_node(node: GraphNode, resources: ResourceManager):
    module = node.module

    if isinstance(module, LLMInference):
        # Resolve alias to actual endpoint
        result = await resources.execute(
            alias=module.alias,
            module=module,
            args=resolved_args,
            kwargs=resolved_kwargs,
        )
```

This separation enables:
- **Environment-specific configs**: Dev uses OpenAI, prod uses self-hosted
- **A/B testing**: Route some requests to new model versions
- **Cost optimization**: Use cheaper models for development
- **Failover**: Switch endpoints without code changes
