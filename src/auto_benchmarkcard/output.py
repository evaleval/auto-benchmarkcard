"""Output directory management for benchmark processing."""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

from auto_benchmarkcard.config import Config

logger = logging.getLogger(__name__)


def sanitize_benchmark_name(name: str) -> str:
    """Convert benchmark name to a canonical, filesystem-safe string.

    All non-alphanumeric characters (except dots for version numbers) are
    replaced with hyphens. Result is lowercase with no leading/trailing
    or consecutive hyphens. Uses hyphens to match Entity Registry conventions.
    """
    s = name.strip().lower()
    s = re.sub(r'[^a-z0-9.]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s


class OutputManager:
    """Manages timestamped output directory structure for benchmark processing."""

    def __init__(self, benchmark_name: str, base_path: Optional[str] = None):
        self.benchmark_name = sanitize_benchmark_name(benchmark_name)
        self.timestamp = self._generate_timestamp()
        self.session_dir = f"{self.benchmark_name}_{self.timestamp}"

        if base_path:
            self.base_dir = os.path.join(base_path, Config.OUTPUT_DIR, self.session_dir)
        else:
            self.base_dir = os.path.join(Config.OUTPUT_DIR, self.session_dir)

        self.tool_output_dir = os.path.join(self.base_dir, Config.TOOL_OUTPUT_DIR)
        self.benchmarkcard_dir = os.path.join(self.base_dir, Config.BENCHMARK_CARD_DIR)

        self._create_directories()

        logger.debug("Output session directory: %s", self.base_dir)

    def _generate_timestamp(self) -> str:
        """Return a human-readable timestamp string."""
        return datetime.now().strftime(Config.TIMESTAMP_FORMAT)

    def _create_directories(self) -> None:
        """Create the standard directory structure."""
        os.makedirs(self.tool_output_dir, exist_ok=True)
        os.makedirs(self.benchmarkcard_dir, exist_ok=True)

    def save_tool_output(self, data: Dict[str, Any], tool_name: str, filename: str) -> str:
        """Save JSON output from a tool and return the file path."""
        tool_dir = os.path.join(self.tool_output_dir, tool_name)
        os.makedirs(tool_dir, exist_ok=True)

        output_file = os.path.join(tool_dir, filename)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        return output_file

    def save_benchmark_card(self, data: Dict[str, Any], filename: str) -> str:
        """Save final benchmark card and return the file path."""
        output_file = os.path.join(self.benchmarkcard_dir, filename)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        return output_file

    def get_tool_output_path(self, tool_name: str, create_if_missing: bool = True) -> str:
        """Return the directory path for a given tool's output."""
        tool_dir = os.path.join(self.tool_output_dir, tool_name)
        if create_if_missing:
            os.makedirs(tool_dir, exist_ok=True)
        return tool_dir

    def get_summary(self) -> Dict[str, str]:
        """Return a summary of output directory paths and timestamp."""
        return {
            "session_directory": self.base_dir,
            "tool_output": self.tool_output_dir,
            "benchmark_cards": self.benchmarkcard_dir,
            "timestamp": self.timestamp,
        }
