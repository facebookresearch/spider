"""Proxy module to allow `from spider.examples.run_mjwp import ...`."""
from importlib import import_module

_run = import_module("examples.run_mjwp")

globals().update({name: getattr(_run, name) for name in dir(_run) if not name.startswith("__")})
