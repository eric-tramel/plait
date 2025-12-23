# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `inf-engine`, a Python project that appears to be in early development stages. The project uses Python 3.13 and includes OpenAI API integration as a core dependency.

## Development Environment

- **Python Version**: 3.13 (specified in `.python-version`)
- **Package Manager**: Uses `uv` for dependency management (presence of `uv.lock`)
- **Virtual Environment**: `.venv` directory (excluded from git)

## Core Dependencies

- `asyncio>=4.0.0` - Async programming support
- `openai>=2.14.0` - OpenAI API client

## Common Commands

### Environment Setup
```bash
# Install dependencies using uv
uv install

# Activate virtual environment (if not using uv run)
source .venv/bin/activate
```

### Running the Application
```bash
# Run the main application
python main.py

# Or using uv
uv run python main.py
```

### Development Tasks
```bash
# Install new dependencies
uv add <package-name>

# Install development dependencies
uv add --dev <package-name>
```

## Project Structure

The project is currently minimal with a basic entry point:
- `main.py` - Main application entry point with a simple "Hello" message
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Locked dependency versions

## Architecture Notes

This appears to be a foundational project setup for an "inference engine" (based on the name). The presence of the OpenAI dependency suggests this will likely involve:
- AI/ML inference capabilities
- API integrations with OpenAI services
- Asynchronous processing (given the asyncio dependency)

The current `main.py` is a placeholder that prints a simple message, indicating the actual application logic is yet to be implemented.