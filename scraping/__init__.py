"""
Scraping module for mining dive log files from GitHub issues.

This module provides tools for:
- Mining dive log attachments from GitHub issue trackers
- Parsing various dive log formats (Subsurface, UDDF, CSV)
- Sanitizing XML data
- Loading profiles for analysis
"""

from scraping.github_miner import GitHubMiner
from scraping.parsers import SubsurfaceParser, UDDFParser, CSVParser
from scraping.sanitizer import sanitize_xml
from scraping.loader import parse_raw_files, load_profiles, profile_to_dict, dict_to_profile

__all__ = [
    "GitHubMiner",
    "SubsurfaceParser",
    "UDDFParser",
    "CSVParser",
    "sanitize_xml",
    "parse_raw_files",
    "load_profiles",
    "profile_to_dict",
    "dict_to_profile",
]
