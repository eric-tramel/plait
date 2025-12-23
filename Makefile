.PHONY: test test-unit test-integration lint types ci

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
