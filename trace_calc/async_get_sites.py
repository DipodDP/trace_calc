"""Helper utilities for file path operations.

This module provides backward-compatible utilities for working with
site coordinate files (.trlc) and path data files (.path).
"""

import os
from pathlib import Path
from .domain.constants import OUTPUT_DATA_DIR


def path_sites(filename: str = "") -> str:
    """
    Get the absolute path to a file in the output data directory.

    Args:
        filename: The filename to join with the output directory path.
                 If empty, returns the output directory path itself.

    Returns:
        Absolute path string to the file or directory.

    Example:
        >>> path_sites("site1 site2.trlc")
        '/home/user/project/output_data/site1 site2.trlc'

        >>> path_sites("")
        '/home/user/project/output_data'
    """
    if filename:
        return str(Path(OUTPUT_DATA_DIR) / filename)
    return OUTPUT_DATA_DIR
