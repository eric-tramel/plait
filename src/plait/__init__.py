"""plait: A PyTorch-inspired framework for LLM inference pipelines.

This package provides tools for building, executing, and optimizing
complex LLM inference pipelines with automatic DAG capture and
maximum throughput through async execution.
"""

from plait.containers import ModuleDict, ModuleList, ParameterDict, ParameterList
from plait.execution.context import ExecutionSettings
from plait.module import LLMInference, Module
from plait.parameter import Parameter

__version__ = "0.1.0"
__all__ = [
    "ExecutionSettings",
    "LLMInference",
    "Module",
    "ModuleDict",
    "ModuleList",
    "Parameter",
    "ParameterDict",
    "ParameterList",
]
