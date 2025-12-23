# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding and development tooling
- Design documentation in `design_docs/`
- Development plan and task breakdown
- Initial package structure with `py.typed` marker (PEP 561)
- Parameter class for learnable string values that can be optimized via backward passes
- InferenceModule base class with automatic child/parameter registration
- InferenceModule introspection methods: `children()`, `modules()`, `parameters()`, and `named_*` variants
- InferenceModule `forward()` abstract method and `__call__()` delegation
- LLMInference atomic module for LLM API calls
- Module composition integration tests
- Practical examples in `examples/` demonstrating modules, parameters, and LLM pipelines
- Trace context infrastructure for DAG capture using ContextVar
- Proxy class for symbolic tracing

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

---

## Version History

_No releases yet._

---

## Release Process

1. Update version in `pyproject.toml`
2. Move items from `[Unreleased]` to new version section
3. Add release date
4. Create git tag: `git tag -a v0.X.0 -m "Release v0.X.0"`
5. Push tag: `git push origin v0.X.0`
