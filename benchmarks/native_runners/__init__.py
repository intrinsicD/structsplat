"""Isolated entry points for native external-reference repositories.

Each runner executes in a subprocess with the reference repository at the front of
``PYTHONPATH``. This prevents repositories that all ship a package named ``gsplat`` from
silently borrowing one another's compiled extension.
"""
