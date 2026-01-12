.PHONY: test test-unit test-integration lint types ci example docs docs-serve doctest

ci: lint types test

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

lint:
	uv run ruff format .
	uv run ruff check --fix .

types:
	uv run ty check

example:
	@for f in examples/[0-9]*.py; do \
		echo "=== Running $$f ==="; \
		uv run python "$$f" || exit 1; \
	done

docs:
	uv run mkdocs build
	cp index.html styles.css public/

docs-serve:
	uv run mkdocs serve

doctest:
	@echo "=== Running plait vs Pydantic AI comparison ==="
	uv run --with pydantic-ai --with rich docs/comparison/compare_pydantic_ai.py
	@echo ""
	@echo "=== Running plait vs LangGraph comparison ==="
	uv run --with langgraph --with langchain-openai --with rich docs/comparison/compare_langgraph.py
	@echo ""
	@echo "=== Running plait vs DSPy comparison ==="
	uv run --with dspy --with rich docs/comparison/compare_dspy.py
