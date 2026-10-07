"""Portable report generation for standard unmixing artifacts.

The report layer deliberately consumes the path based :class:`UnmixArtifact`
contract rather than importing a solver implementation. This keeps plotting
and evaluation reproducible when the optimisation engine changes. A report
set contains paper-style sample unmixing plots, an active-phase recovery
matrix, an anomaly ranking, and their supporting tables.
"""

from .render import render_reports

__all__ = ["render_reports"]
