"""Entry point for running auto_benchmarkcard as a module.

This allows the package to be run as:
    python -m auto_benchmarkcard <command> [options]

Commands: generate, generate-unitxt, list, show, validate.
"""

import sys

from auto_benchmarkcard.cli import app

if __name__ == "__main__":
    app()
