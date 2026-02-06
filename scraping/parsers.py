"""Parse dive log files into DiveProfile objects."""

import csv
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Dict

from backtest.profile_generator import DiveProfile

logger = logging.getLogger(__name__)


def _parse_time_str(time_str: str) -> float:
    """
    Convert time string to minutes.

    Supports formats:
    - "MM:SS min"
    - "HH:MM:SS"
    - "MM.SS"
    - Plain number strings

    Args:
        time_str: Time string to parse

    Returns:
        Time in minutes
    """
    time_str = time_str.strip().lower().replace('min', '').strip()

    # Try HH:MM:SS or MM:SS
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 60 + minutes + seconds / 60
        elif len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes + seconds / 60

    # Try plain number
    return float(time_str)


def _parse_depth_str(depth_str: str) -> float:
    """
    Convert depth string to meters.

    Supports formats:
    - "XX.X m"
    - "XX.X meters"
    - "XX.X ft" (converts to meters)
    - Plain number strings

    Args:
        depth_str: Depth string to parse

    Returns:
        Depth in meters
    """
    depth_str = depth_str.strip().lower()

    # Extract number
    match = re.search(r'([\d.]+)', depth_str)
    if not match:
        return 0.0

    value = float(match.group(1))

    # Check for feet
    if 'ft' in depth_str or 'feet' in depth_str:
        value *= 0.3048  # Convert to meters

    return value


def _parse_percentage(percent_str: str) -> float:
    """
    Convert percentage string to fraction.

    Supports formats:
    - "32.0%"
    - "0.32"
    - "32"

    Args:
        percent_str: Percentage string

    Returns:
        Fraction (0.0-1.0)
    """
    percent_str = percent_str.strip().replace('%', '')
    value = float(percent_str)

    if value > 1.0:
        value /= 100.0

    return value


def _calculate_bottom_time(points: list, depth_threshold: float = 3.0) -> float:
    """Calculate time spent deeper than threshold by summing qualifying segments."""
    bottom_time = 0.0
    for i in range(len(points) - 1):
        t1, d1 = points[i][0], points[i][1]
        t2, d2 = points[i + 1][0], points[i + 1][1]
        if d1 > depth_threshold or d2 > depth_threshold:
            bottom_time += t2 - t1
    return bottom_time


class SubsurfaceParser:
    """Parse Subsurface XML dive log files."""

    def parse_file(self, filepath: str) -> List[DiveProfile]:
        """
        Parse a Subsurface XML file.

        Args:
            filepath: Path to XML file

        Returns:
            List of DiveProfile objects
        """
        try:
            content = Path(filepath).read_text(encoding='utf-8', errors='replace')
            return self.parse_string(content)
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            return []

    def parse_string(self, content: str) -> List[DiveProfile]:
        """
        Parse Subsurface XML content.

        Args:
            content: XML string

        Returns:
            List of DiveProfile objects
        """
        profiles = []

        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            return []

        # Find all dive elements (may be nested in trips)
        for dive_elem in root.findall('.//dive'):
            try:
                profile = self._parse_dive_element(dive_elem)
                if profile:
                    profiles.append(profile)
            except Exception as e:
                dive_num = dive_elem.get('number', 'unknown')
                logger.warning(f"Failed to parse dive {dive_num}: {e}")

        logger.info(f"Parsed {len(profiles)} dives from Subsurface XML")
        return profiles

    def _parse_dive_element(self, dive_elem: ET.Element) -> Optional[DiveProfile]:
        """Parse a single dive element."""
        dive_num = dive_elem.get('number', 'unknown')
        date = dive_elem.get('date', '')
        time = dive_elem.get('time', '')

        name = f"dive_{dive_num}"
        if date:
            name = f"dive_{date}_{dive_num}"

        profile = DiveProfile(name=name)

        # Get gas mix from cylinders
        fO2 = 0.21  # Default to air
        fHe = 0.0

        cylinder = dive_elem.find('.//cylinder')
        if cylinder is not None:
            o2_str = cylinder.get('o2', '21.0%')
            he_str = cylinder.get('he', '0.0%')
            try:
                fO2 = _parse_percentage(o2_str)
                fHe = _parse_percentage(he_str)
            except ValueError:
                pass

        # Parse samples from divecomputer
        divecomputer = dive_elem.find('.//divecomputer')
        if divecomputer is None:
            logger.warning(f"No divecomputer data in dive {dive_num}")
            return None

        samples = divecomputer.findall('sample')
        if not samples:
            logger.warning(f"No samples in dive {dive_num}")
            return None

        current_fO2 = fO2
        current_fHe = fHe

        for sample in samples:
            try:
                time_str = sample.get('time', '0:00 min')
                depth_str = sample.get('depth', '0.0 m')

                time_min = _parse_time_str(time_str)
                depth_m = _parse_depth_str(depth_str)

                # Check for gas change events
                for event in divecomputer.findall('event'):
                    event_time = _parse_time_str(event.get('time', '0:00 min'))
                    if abs(event_time - time_min) < 0.01:  # Same timestamp
                        if event.get('type') == '25' or event.get('name') == 'gaschange':
                            # Gas change event - would need to look up cylinder
                            # For now, keep current gas
                            pass

                profile.add_point(time_min, depth_m, current_fO2, current_fHe)

            except (ValueError, AttributeError) as e:
                logger.debug(f"Skipping malformed sample: {e}")

        if len(profile.points) < 2:
            return None

        profile.bottom_time = _calculate_bottom_time(profile.points)

        return profile


class UDDFParser:
    """Parse UDDF (Universal Dive Data Format) XML files."""

    def parse_file(self, filepath: str) -> List[DiveProfile]:
        """
        Parse a UDDF XML file.

        Args:
            filepath: Path to XML file

        Returns:
            List of DiveProfile objects
        """
        try:
            content = Path(filepath).read_text(encoding='utf-8', errors='replace')
            return self.parse_string(content)
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            return []

    def parse_string(self, content: str) -> List[DiveProfile]:
        """
        Parse UDDF XML content.

        Args:
            content: XML string

        Returns:
            List of DiveProfile objects
        """
        profiles = []

        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse UDDF XML: {e}")
            return []

        # Parse gas definitions
        gas_mixes = self._parse_gas_definitions(root)

        # Find all dives
        dive_num = 0
        for dive_elem in root.findall('.//dive'):
            dive_num += 1
            try:
                profile = self._parse_dive_element(dive_elem, dive_num, gas_mixes)
                if profile:
                    profiles.append(profile)
            except Exception as e:
                logger.warning(f"Failed to parse UDDF dive {dive_num}: {e}")

        logger.info(f"Parsed {len(profiles)} dives from UDDF")
        return profiles

    def _parse_gas_definitions(self, root: ET.Element) -> Dict[str, tuple]:
        """Parse gas mix definitions."""
        gas_mixes = {}

        gasdefs = root.find('.//gasdefinitions')
        if gasdefs is None:
            return gas_mixes

        for mix in gasdefs.findall('mix'):
            mix_id = mix.get('id')
            if not mix_id:
                continue

            o2_elem = mix.find('o2')
            n2_elem = mix.find('n2')
            he_elem = mix.find('he')

            fO2 = float(o2_elem.text) if o2_elem is not None and o2_elem.text else 0.21
            fHe = float(he_elem.text) if he_elem is not None and he_elem.text else 0.0

            gas_mixes[mix_id] = (fO2, fHe)

        return gas_mixes

    def _parse_dive_element(
        self, dive_elem: ET.Element, dive_num: int, gas_mixes: Dict[str, tuple]
    ) -> Optional[DiveProfile]:
        """Parse a single UDDF dive element."""
        name = f"uddf_dive_{dive_num}"
        profile = DiveProfile(name=name)

        # Default gas
        fO2 = 0.21
        fHe = 0.0

        # Find samples
        samples_elem = dive_elem.find('.//samples')
        if samples_elem is None:
            logger.warning(f"No samples in UDDF dive {dive_num}")
            return None

        waypoints = samples_elem.findall('waypoint')
        if not waypoints:
            logger.warning(f"No waypoints in UDDF dive {dive_num}")
            return None

        for waypoint in waypoints:
            try:
                divetime_elem = waypoint.find('divetime')
                depth_elem = waypoint.find('depth')

                if divetime_elem is None or depth_elem is None:
                    continue

                # UDDF divetime is in seconds
                time_sec = float(divetime_elem.text or "0")
                time_min = time_sec / 60.0

                depth_m = float(depth_elem.text or "0")

                # Check for gas switch
                switchmix = waypoint.find('switchmix')
                if switchmix is not None:
                    mix_ref = switchmix.get('ref')
                    if mix_ref and mix_ref in gas_mixes:
                        fO2, fHe = gas_mixes[mix_ref]

                profile.add_point(time_min, depth_m, fO2, fHe)

            except (ValueError, AttributeError) as e:
                logger.debug(f"Skipping malformed waypoint: {e}")

        if len(profile.points) < 2:
            return None

        profile.bottom_time = _calculate_bottom_time(profile.points)

        return profile


class CSVParser:
    """Parse CSV dive log files with flexible column detection."""

    # Column name patterns
    TIME_COLUMNS = ['time', 'time_min', 'elapsed_time', 'divetime', 'runtime']
    DEPTH_COLUMNS = ['depth', 'depth_m', 'depth_meters']
    O2_COLUMNS = ['o2', 'fO2', 'o2_fraction', 'oxygen']
    HE_COLUMNS = ['he', 'fHe', 'he_fraction', 'helium']

    def parse_file(self, filepath: str) -> List[DiveProfile]:
        """
        Parse a CSV dive log file.

        Args:
            filepath: Path to CSV file

        Returns:
            List of DiveProfile objects (usually just one)
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if not rows:
                    logger.warning(f"Empty CSV file: {filepath}")
                    return []

                # Auto-detect columns
                columns = self._detect_columns(rows[0].keys())

                if not columns['time'] or not columns['depth']:
                    # Try to create square profile from metadata
                    return self._try_metadata_profile(rows, filepath)

                return self._parse_timeseries(rows, columns, filepath)

        except Exception as e:
            logger.error(f"Failed to parse CSV {filepath}: {e}")
            return []

    def _detect_columns(self, headers) -> Dict[str, Optional[str]]:
        """Auto-detect column names."""
        headers_lower = {h.lower(): h for h in headers}

        columns: Dict[str, Optional[str]] = {
            'time': None,
            'depth': None,
            'o2': None,
            'he': None,
        }

        # Find time column
        for pattern in self.TIME_COLUMNS:
            if pattern in headers_lower:
                columns['time'] = headers_lower[pattern]
                break

        # Find depth column
        for pattern in self.DEPTH_COLUMNS:
            if pattern in headers_lower:
                columns['depth'] = headers_lower[pattern]
                break

        # Find O2 column
        for pattern in self.O2_COLUMNS:
            if pattern in headers_lower:
                columns['o2'] = headers_lower[pattern]
                break

        # Find He column
        for pattern in self.HE_COLUMNS:
            if pattern in headers_lower:
                columns['he'] = headers_lower[pattern]
                break

        return columns

    def _parse_timeseries(
        self, rows: List[Dict[str, str]], columns: Dict[str, Optional[str]], filepath: str
    ) -> List[DiveProfile]:
        """Parse time-series CSV data."""
        name = Path(filepath).stem
        profile = DiveProfile(name=name)

        fO2 = 0.21
        fHe = 0.0

        time_col = columns['time']
        depth_col = columns['depth']
        o2_col = columns['o2']
        he_col = columns['he']

        for row in rows:
            try:
                time_str = row[time_col]  # type: ignore[index]
                depth_str = row[depth_col]  # type: ignore[index]

                time_min = _parse_time_str(time_str)
                depth_m = _parse_depth_str(depth_str)

                # Get gas fractions if available
                if o2_col and o2_col in row:
                    fO2 = _parse_percentage(row[o2_col])
                if he_col and he_col in row:
                    fHe = _parse_percentage(row[he_col])

                profile.add_point(time_min, depth_m, fO2, fHe)

            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping malformed CSV row: {e}")

        if len(profile.points) < 2:
            logger.warning(f"Insufficient points in CSV {filepath}")
            return []

        profile.bottom_time = _calculate_bottom_time(profile.points)

        return [profile]

    def _try_metadata_profile(
        self, rows: List[Dict[str, str]], filepath: str
    ) -> List[DiveProfile]:
        """Try to create square profile from max_depth and bottom_time columns."""
        headers_lower = {k.lower(): k for k in rows[0].keys()}

        if 'max_depth' not in headers_lower or 'bottom_time' not in headers_lower:
            logger.warning(f"No time-series or metadata columns in CSV {filepath}")
            return []

        name = Path(filepath).stem
        profile = DiveProfile(name=name)

        # Get first row metadata
        row = rows[0]
        max_depth = _parse_depth_str(row[headers_lower['max_depth']])
        bottom_time = _parse_time_str(row[headers_lower['bottom_time']])

        # Create simple square profile
        # Descent at 20m/min
        descent_time = max_depth / 20.0

        profile.add_point(0.0, 0.0)
        profile.add_point(descent_time, max_depth)
        profile.add_point(descent_time + bottom_time, max_depth)
        profile.add_point(descent_time + bottom_time + max_depth / 10.0, 0.0)

        profile.bottom_time = bottom_time

        logger.info(f"Created square profile from CSV metadata: {max_depth}m, {bottom_time}min")
        return [profile]
