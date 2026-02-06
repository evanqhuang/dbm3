"""Load scraped and parsed dive profiles for backtesting."""

import gzip
import json
import logging
import zipfile
from pathlib import Path
from typing import List, Dict, Any

from backtest.profile_generator import DiveProfile
from scraping.parsers import SubsurfaceParser, UDDFParser, CSVParser
from scraping.sanitizer import sanitize_xml

logger = logging.getLogger(__name__)

# Map file extensions to parsers
PARSER_MAP = {
    '.xml': SubsurfaceParser,
    '.ssrf': SubsurfaceParser,
    '.uddf': UDDFParser,
    '.csv': CSVParser,
}


def _extract_zips(raw_path: Path) -> None:
    """Extract all ZIP archives, placing contents in a sibling .extracted directory.

    Idempotent: skips ZIPs that have already been extracted.
    Secure: rejects entries with path traversal (.. or absolute paths).
    """
    for zip_file in raw_path.rglob("*.zip"):
        extract_dir = zip_file.with_suffix('.extracted')
        if extract_dir.exists():
            continue

        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                safe_members = [
                    m for m in zf.namelist()
                    if not m.startswith('/') and '..' not in m
                ]
                extract_dir.mkdir(parents=True, exist_ok=True)
                for member in safe_members:
                    zf.extract(member, extract_dir)
            logger.info(f"Extracted {len(safe_members)} files from {zip_file.name}")
        except (zipfile.BadZipFile, Exception) as e:
            logger.warning(f"Failed to extract {zip_file.name}: {e}")


def _try_gzip_parse(filepath: Path) -> List[DiveProfile]:
    """Try to parse a file as gzip-compressed Subsurface XML or UDDF."""
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(2)
        if magic != b'\x1f\x8b':
            return []

        content = gzip.open(filepath, 'rt', encoding='utf-8', errors='replace').read()
        content = sanitize_xml(content)

        # Try Subsurface XML first, then UDDF
        profiles = SubsurfaceParser().parse_string(content)
        if profiles:
            return profiles
        return UDDFParser().parse_string(content)
    except Exception as e:
        logger.debug(f"Gzip parse failed for {filepath.name}: {e}")
        return []


def parse_raw_files(raw_dir: str = "data/raw", parsed_dir: str = "data/parsed") -> int:
    """
    Parse all raw dive log files and save as JSON.

    Args:
        raw_dir: Directory containing raw dive log files
        parsed_dir: Directory to save parsed JSON files

    Returns:
        Number of profiles parsed
    """
    raw_path = Path(raw_dir)
    parsed_path = Path(parsed_dir)

    if not raw_path.exists():
        logger.warning(f"Raw data directory does not exist: {raw_dir}")
        return 0

    parsed_path.mkdir(parents=True, exist_ok=True)

    # Extract ZIP archives first so their contents are available for parsing
    _extract_zips(raw_path)

    count = 0
    for filepath in raw_path.rglob("*"):
        if not filepath.is_file():
            continue

        ext = filepath.suffix.lower()

        # Try gzip detection for .bin/.gz files
        if ext in ('.bin', '.gz'):
            output_file = parsed_path / f"{filepath.stem}.json"
            if output_file.exists():
                continue
            profiles = _try_gzip_parse(filepath)
            if profiles:
                data = [{"name": p.name, "max_depth": p.max_depth,
                         "bottom_time": p.bottom_time, "points": p.points} for p in profiles]
                output_file.write_text(json.dumps(data, indent=2))
                count += len(profiles)
                logger.info(f"Parsed {len(profiles)} profiles from gzip {filepath.name}")
            continue

        if ext not in PARSER_MAP:
            continue

        # Skip if already parsed
        output_file = parsed_path / f"{filepath.stem}.json"
        if output_file.exists():
            logger.debug(f"Skipping already parsed file: {filepath.name}")
            continue

        try:
            parser = PARSER_MAP[ext]()

            # Sanitize XML files before parsing
            if ext in ('.xml', '.ssrf', '.uddf'):
                content = filepath.read_text(encoding='utf-8', errors='replace')
                content = sanitize_xml(content)
                profiles = parser.parse_string(content)
            else:
                profiles = parser.parse_file(str(filepath))

            if profiles:
                # Save as JSON
                data = []
                for p in profiles:
                    data.append({
                        "name": p.name,
                        "max_depth": p.max_depth,
                        "bottom_time": p.bottom_time,
                        "points": p.points,
                    })
                output_file.write_text(json.dumps(data, indent=2))
                count += len(profiles)
                logger.info(f"Parsed {len(profiles)} profiles from {filepath.name}")

        except Exception as e:
            logger.warning(f"Failed to parse {filepath.name}: {e}")

    logger.info(f"Total parsed: {count} profiles")
    return count


def load_profiles(
    data_dir: str = "data/parsed",
    min_points: int = 10,
    max_depth: float = 200.0,
    min_depth: float = 3.0,
) -> List[DiveProfile]:
    """
    Load all parsed profiles, filtering out invalid or sparse ones.

    Args:
        data_dir: Directory containing parsed JSON files
        min_points: Minimum number of sample points to include
        max_depth: Maximum depth to consider valid (meters)
        min_depth: Minimum depth to consider valid (meters)

    Returns:
        List of DiveProfile objects ready for backtesting
    """
    parsed_path = Path(data_dir)
    if not parsed_path.exists():
        logger.warning(f"No parsed data directory: {data_dir}")
        return []

    profiles = []
    for json_file in sorted(parsed_path.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding='utf-8'))

            for entry in data:
                profile = DiveProfile(
                    name=entry["name"],
                    max_depth=entry["max_depth"],
                    bottom_time=entry["bottom_time"],
                    points=[tuple(p) for p in entry["points"]],
                )

                # Filter invalid profiles
                if len(profile.points) < min_points:
                    logger.debug(f"Skipping {profile.name}: too few points ({len(profile.points)})")
                    continue

                if profile.max_depth > max_depth:
                    logger.debug(f"Skipping {profile.name}: too deep ({profile.max_depth}m)")
                    continue

                if profile.max_depth < min_depth:
                    logger.debug(f"Skipping {profile.name}: too shallow ({profile.max_depth}m)")
                    continue

                profiles.append(profile)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load {json_file.name}: {e}")

    logger.info(f"Loaded {len(profiles)} valid profiles from {data_dir}")
    return profiles


def profile_to_dict(profile: DiveProfile) -> Dict[str, Any]:
    """
    Convert a DiveProfile to a JSON-serializable dictionary.

    Args:
        profile: DiveProfile object

    Returns:
        Dictionary representation
    """
    return {
        "name": profile.name,
        "max_depth": profile.max_depth,
        "bottom_time": profile.bottom_time,
        "points": profile.points,
    }


def dict_to_profile(data: Dict[str, Any]) -> DiveProfile:
    """
    Convert a dictionary to a DiveProfile object.

    Args:
        data: Dictionary representation

    Returns:
        DiveProfile object
    """
    return DiveProfile(
        name=data["name"],
        max_depth=data["max_depth"],
        bottom_time=data["bottom_time"],
        points=[tuple(p) for p in data["points"]],
    )
