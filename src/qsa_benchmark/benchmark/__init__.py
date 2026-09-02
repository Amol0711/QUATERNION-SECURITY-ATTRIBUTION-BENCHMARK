"""Controlled benchmark architecture for quaternion security attribution."""

from .registry import METHOD_FACTORIES, make_method, method_registry

__all__ = ["METHOD_FACTORIES", "make_method", "method_registry"]
