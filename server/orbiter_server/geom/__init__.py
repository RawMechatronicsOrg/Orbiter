"""Geometry math for Orbiter — Python port of the frontend `ui/src/scanner/math/`.

The server is the authoritative holder of model state and does all geometry
math (Viser-pattern migration). These modules are pure: numpy/scipy only, no
FastAPI imports, so they are unit-testable in isolation.

Ground truth: COORDINATES.md §5 (camera-pose formulas, Z-up world frame).
"""
