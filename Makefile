.PHONY: test test-unit test-integration lint types ci example docs docs-serve doctest

ci: lint types test

test:
	uv run --group dev -m pytest

test-unit:
	uv run --group dev -m pytest tests/unit

test-integration:
	uv run --group dev -m pytest tests/integration

lint:
	uv run --group dev ruff format .
	uv run --group dev ruff check --fix .

types:
	uv run --group dev ty check

example:
	@for f in examples/[0-9]*.py; do \
		echo "=== Running $$f ==="; \
		uv run python "$$f" || exit 1; \
	done

docs:
	uv run --group dev mkdocs build
	cp index.html styles.css public/

docs-serve:
	uv run --group dev mkdocs serve

doctest:
	@echo "=== Running plait vs Pydantic AI comparison ==="
	uv run --with pydantic-ai --with rich docs/comparison/compare_pydantic_ai.py
	@echo ""
	@echo "=== Running plait vs LangGraph comparison ==="
	uv run --with langgraph --with langchain-openai --with rich docs/comparison/compare_langgraph.py
	@echo ""
	@echo "=== Running plait vs DSPy comparison ==="
	uv run --with dspy --with rich docs/comparison/compare_dspy.py
