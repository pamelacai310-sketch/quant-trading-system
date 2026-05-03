"""
Integrations module for external system bridges.

Provides bridge integrations with external platforms and tools including
FinceptTerminal and other financial platforms.
"""

from .fincept_bridge import (
    FinceptConfig,
    FinceptDataBridge,
    FinceptStrategyExporter,
    FinceptSignalPusher,
    FinceptIntegrator,
    create_fincept_integration,
)

__all__ = [
    "FinceptConfig",
    "FinceptDataBridge",
    "FinceptStrategyExporter",
    "FinceptSignalPusher",
    "FinceptIntegrator",
    "create_fincept_integration",
]
