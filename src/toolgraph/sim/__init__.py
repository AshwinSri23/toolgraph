"""Deterministic simulation: virtual time, fake MCP servers, crash injection."""

from .harness import NodeSpec, Scenario, SimulatedCrash, check, crash_points, run_scenario
from .vloop import SimulationDeadlock, VirtualTimeLoop, run_virtual
from .world import SimBackend, SimTool, World

__all__ = [
    "NodeSpec",
    "Scenario",
    "SimBackend",
    "SimTool",
    "SimulatedCrash",
    "SimulationDeadlock",
    "VirtualTimeLoop",
    "World",
    "check",
    "crash_points",
    "run_scenario",
    "run_virtual",
]
