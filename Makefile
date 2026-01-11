.PHONY: test test-unit test-integration lint types ci example

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
