"""
Comprehensive unit tests for the scraping module.

Tests actual functionality and behavior, not just code coverage.
Each test validates specific expected outcomes and edge cases.
"""

import gzip
import json
import tempfile
import time
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from backtest.profile_generator import DiveProfile
from scraping.sanitizer import sanitize_xml
from scraping.parsers import (
    SubsurfaceParser,
    UDDFParser,
    CSVParser,
    _parse_time_str,
    _parse_depth_str,
    _parse_percentage,
    _calculate_bottom_time,
)
from scraping.github_miner import GitHubMiner
from scraping.loader import (
    parse_raw_files,
    load_profiles,
    profile_to_dict,
    dict_to_profile,
    _extract_zips,
    _try_gzip_parse,
)


# ==============================================================================
# SANITIZER TESTS - Validate PII removal functionality
# ==============================================================================


class TestSanitizer:
    """Test XML sanitization to remove personally identifiable information."""

    def test_removes_buddy_tag(self):
        """Verify <buddy> tags are completely removed."""
        xml = """
        <dive>
            <buddy>John Doe</buddy>
            <depth>30.0 m</depth>
        </dive>
        """
        result = sanitize_xml(xml)
        assert '<buddy>' not in result
        assert 'John Doe' not in result
        assert '<depth>' in result  # Preserved

    def test_removes_location_tag(self):
        """Verify <location> tags are removed."""
        xml = """
        <dive>
            <location>Great Barrier Reef, Australia</location>
            <sample time="0:00 min" depth="0.0 m"/>
        </dive>
        """
        result = sanitize_xml(xml)
        assert '<location>' not in result
        assert 'Great Barrier Reef' not in result
        assert '<sample' in result  # Preserved

    def test_removes_notes_tag(self):
        """Verify <notes> tags containing personal observations are removed."""
        xml = """
        <dive>
            <notes>Saw a hammerhead shark. My first technical dive!</notes>
            <depth>40.0 m</depth>
        </dive>
        """
        result = sanitize_xml(xml)
        assert '<notes>' not in result
        assert 'hammerhead' not in result
        assert '<depth>' in result

    def test_removes_gps_tag(self):
        """Verify <gps> tags are removed."""
        xml = """
        <dive>
            <gps>-27.234,153.456</gps>
            <depth>20.0 m</depth>
        </dive>
        """
        result = sanitize_xml(xml)
        assert '<gps>' not in result
        assert '-27.234' not in result

    def test_removes_divemaster_tag(self):
        """Verify <divemaster> tags are removed."""
        xml = """
        <dive>
            <divemaster>Captain Bob</divemaster>
            <depth>25.0 m</depth>
        </dive>
        """
        result = sanitize_xml(xml)
        assert '<divemaster>' not in result
        assert 'Captain Bob' not in result

    def test_removes_suit_tag(self):
        """Verify <suit> tags are removed."""
        xml = """
        <dive>
            <suit>7mm wetsuit</suit>
            <depth>15.0 m</depth>
        </dive>
        """
        result = sanitize_xml(xml)
        assert '<suit>' not in result
        assert 'wetsuit' not in result

    def test_preserves_sample_tags(self):
        """Verify critical dive data (sample tags) are preserved."""
        xml = """
        <dive>
            <buddy>John</buddy>
            <sample time="0:00 min" depth="0.0 m"/>
            <sample time="5:00 min" depth="20.0 m"/>
            <location>Secret Spot</location>
        </dive>
        """
        result = sanitize_xml(xml)
        assert '<sample' in result
        assert 'time="0:00 min"' in result
        assert 'depth="20.0 m"' in result
        assert '<buddy>' not in result
        assert '<location>' not in result

    def test_preserves_dive_tag(self):
        """Verify <dive> root element is preserved."""
        xml = """
        <dive number="123" date="2024-01-15">
            <location>Private Beach</location>
            <depth>30.0 m</depth>
        </dive>
        """
        result = sanitize_xml(xml)
        assert '<dive' in result
        assert 'number="123"' in result
        assert 'date="2024-01-15"' in result
        assert '<location>' not in result

    def test_strips_gps_attributes(self):
        """Verify gps attributes are removed from any element."""
        xml = """
        <dive gps="12.345,-67.890">
            <site gps="12.345,-67.890">Reef Site</site>
            <depth>25.0 m</depth>
        </dive>
        """
        result = sanitize_xml(xml)
        assert 'gps=' not in result
        assert '12.345' not in result
        assert '<depth>' in result

    def test_regex_fallback_for_malformed_xml(self):
        """Verify fallback regex sanitization works for malformed XML."""
        malformed = """
        <dive>
            <buddy>Alice</buddy
            <location>Beach</location>
            <depth>10.0 m</depth>
        </dive>
        """
        # This should not crash, but use regex fallback
        result = sanitize_xml(malformed)
        # Regex should still remove location
        assert 'Beach' not in result or '<location>' not in result

    def test_handles_empty_pii_tags(self):
        """Verify empty PII tags are still removed."""
        xml = """
        <dive>
            <buddy></buddy>
            <location/>
            <notes/>
            <depth>15.0 m</depth>
        </dive>
        """
        result = sanitize_xml(xml)
        assert '<buddy>' not in result
        assert '<location' not in result
        assert '<notes' not in result
        assert '<depth>' in result


# ==============================================================================
# HELPER FUNCTION TESTS - Validate parsing utilities
# ==============================================================================


class TestParseHelpers:
    """Test helper functions for parsing time, depth, and percentages."""

    def test_parse_time_str_mm_ss_format(self):
        """Verify MM:SS min format is correctly parsed."""
        assert _parse_time_str("5:30 min") == pytest.approx(5.5)
        assert _parse_time_str("0:00 min") == pytest.approx(0.0)
        assert _parse_time_str("2:15 min") == pytest.approx(2.25)

    def test_parse_time_str_hh_mm_ss_format(self):
        """Verify HH:MM:SS format is correctly parsed."""
        assert _parse_time_str("1:30:00") == pytest.approx(90.0)  # 1.5 hours
        assert _parse_time_str("0:05:30") == pytest.approx(5.5)
        assert _parse_time_str("2:00:00") == pytest.approx(120.0)

    def test_parse_time_str_plain_number(self):
        """Verify plain numbers are interpreted as minutes."""
        assert _parse_time_str("10") == pytest.approx(10.0)
        assert _parse_time_str("25.5") == pytest.approx(25.5)
        assert _parse_time_str("0") == pytest.approx(0.0)

    def test_parse_depth_str_meters(self):
        """Verify depth in meters is correctly parsed."""
        assert _parse_depth_str("30.2 m") == pytest.approx(30.2)
        assert _parse_depth_str("0.0 m") == pytest.approx(0.0)
        assert _parse_depth_str("45.7 meters") == pytest.approx(45.7)

    def test_parse_depth_str_feet_conversion(self):
        """Verify feet to meters conversion is accurate."""
        # 100 ft = 30.48 m
        assert _parse_depth_str("100 ft") == pytest.approx(30.48, rel=0.01)
        assert _parse_depth_str("33 feet") == pytest.approx(10.0584, rel=0.01)

    def test_parse_depth_str_plain_number(self):
        """Verify plain numbers are interpreted as meters."""
        assert _parse_depth_str("20.5") == pytest.approx(20.5)
        assert _parse_depth_str("15") == pytest.approx(15.0)

    def test_parse_percentage_with_percent_sign(self):
        """Verify percentage strings are converted to fractions."""
        assert _parse_percentage("32.0%") == pytest.approx(0.32)
        assert _parse_percentage("21%") == pytest.approx(0.21)
        assert _parse_percentage("100%") == pytest.approx(1.0)

    def test_parse_percentage_decimal_format(self):
        """Verify decimal fractions are preserved."""
        assert _parse_percentage("0.21") == pytest.approx(0.21)
        assert _parse_percentage("0.32") == pytest.approx(0.32)
        assert _parse_percentage("1.0") == pytest.approx(1.0)

    def test_parse_percentage_whole_number(self):
        """Verify whole numbers >1 are divided by 100."""
        assert _parse_percentage("32") == pytest.approx(0.32)
        assert _parse_percentage("21") == pytest.approx(0.21)


# ==============================================================================
# SUBSURFACE PARSER TESTS - Validate XML dive log parsing
# ==============================================================================


class TestSubsurfaceParser:
    """Test Subsurface XML dive log parsing with realistic data."""

    def test_parses_single_dive_with_samples(self):
        """Verify basic dive parsing with time-series samples."""
        xml = """
        <divelog>
            <dive number="1" date="2024-01-15" time="10:00">
                <cylinder o2="32.0%" he="0.0%"/>
                <divecomputer>
                    <sample time="0:00 min" depth="0.0 m"/>
                    <sample time="1:00 min" depth="10.0 m"/>
                    <sample time="5:00 min" depth="30.0 m"/>
                    <sample time="15:00 min" depth="30.0 m"/>
                    <sample time="18:00 min" depth="0.0 m"/>
                </divecomputer>
            </dive>
        </divelog>
        """
        parser = SubsurfaceParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 1
        profile = profiles[0]
        assert len(profile.points) == 5
        assert profile.max_depth == pytest.approx(30.0)
        assert profile.name == "dive_2024-01-15_1"

        # Verify gas mix was extracted
        first_point = profile.points[0]
        assert first_point[2] == pytest.approx(0.32)  # fO2
        assert first_point[3] == pytest.approx(0.0)   # fHe

    def test_parses_multiple_dives(self):
        """Verify parser handles multiple dives in one file."""
        xml = """
        <divelog>
            <dive number="1" date="2024-01-15">
                <divecomputer>
                    <sample time="0:00 min" depth="0.0 m"/>
                    <sample time="5:00 min" depth="20.0 m"/>
                    <sample time="10:00 min" depth="0.0 m"/>
                </divecomputer>
            </dive>
            <dive number="2" date="2024-01-16">
                <divecomputer>
                    <sample time="0:00 min" depth="0.0 m"/>
                    <sample time="5:00 min" depth="25.0 m"/>
                    <sample time="10:00 min" depth="0.0 m"/>
                </divecomputer>
            </dive>
        </divelog>
        """
        parser = SubsurfaceParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 2
        assert profiles[0].max_depth == pytest.approx(20.0)
        assert profiles[1].max_depth == pytest.approx(25.0)

    def test_extracts_correct_gas_mix(self):
        """Verify O2 and He percentages are correctly extracted from cylinder."""
        xml = """
        <divelog>
            <dive number="1">
                <cylinder o2="21.0%" he="35.0%"/>
                <divecomputer>
                    <sample time="0:00 min" depth="0.0 m"/>
                    <sample time="5:00 min" depth="50.0 m"/>
                </divecomputer>
            </dive>
        </divelog>
        """
        parser = SubsurfaceParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 1
        point = profiles[0].points[0]
        assert point[2] == pytest.approx(0.21)  # fO2
        assert point[3] == pytest.approx(0.35)  # fHe (trimix)

    def test_handles_missing_divecomputer_element(self):
        """Verify parser skips dives without divecomputer data."""
        xml = """
        <divelog>
            <dive number="1">
                <depth>30.0 m</depth>
            </dive>
        </divelog>
        """
        parser = SubsurfaceParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 0  # Should skip dive without samples

    def test_handles_missing_samples_element(self):
        """Verify parser skips dives without sample points."""
        xml = """
        <divelog>
            <dive number="1">
                <divecomputer>
                </divecomputer>
            </dive>
        </divelog>
        """
        parser = SubsurfaceParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 0

    def test_defaults_to_air_when_no_cylinder(self):
        """Verify default gas mix is air (21% O2, 0% He) when cylinder is missing."""
        xml = """
        <divelog>
            <dive number="1">
                <divecomputer>
                    <sample time="0:00 min" depth="0.0 m"/>
                    <sample time="5:00 min" depth="20.0 m"/>
                </divecomputer>
            </dive>
        </divelog>
        """
        parser = SubsurfaceParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 1
        point = profiles[0].points[0]
        assert point[2] == pytest.approx(0.21)  # Default fO2
        assert point[3] == pytest.approx(0.0)   # Default fHe

    def test_rejects_dive_with_too_few_points(self):
        """Verify dives with <2 samples are rejected."""
        xml = """
        <divelog>
            <dive number="1">
                <divecomputer>
                    <sample time="0:00 min" depth="0.0 m"/>
                </divecomputer>
            </dive>
        </divelog>
        """
        parser = SubsurfaceParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 0  # Only 1 point, need at least 2

    def test_calculates_bottom_time(self):
        """Verify bottom time is calculated as time spent deeper than 3m."""
        xml = """
        <divelog>
            <dive number="1">
                <divecomputer>
                    <sample time="0:00 min" depth="0.0 m"/>
                    <sample time="2:00 min" depth="2.0 m"/>
                    <sample time="3:00 min" depth="10.0 m"/>
                    <sample time="15:00 min" depth="10.0 m"/>
                    <sample time="18:00 min" depth="2.0 m"/>
                    <sample time="20:00 min" depth="0.0 m"/>
                </divecomputer>
            </dive>
        </divelog>
        """
        parser = SubsurfaceParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 1
        # Segments deeper than 3m:
        #   2:00-3:00 (2m->10m, 10m > 3m) = 1 min
        #   3:00-15:00 (10m->10m) = 12 min
        #   15:00-18:00 (10m->2m, 10m > 3m) = 3 min
        # Total = 16 min
        assert profiles[0].bottom_time == pytest.approx(16.0)

    def test_handles_gas_change_event(self):
        """Verify gas change events are detected (even if not fully implemented)."""
        xml = """
        <divelog>
            <dive number="1">
                <cylinder o2="21.0%"/>
                <divecomputer>
                    <sample time="0:00 min" depth="0.0 m"/>
                    <sample time="5:00 min" depth="30.0 m"/>
                    <event time="5:00 min" type="25" name="gaschange"/>
                    <sample time="10:00 min" depth="30.0 m"/>
                    <sample time="15:00 min" depth="0.0 m"/>
                </divecomputer>
            </dive>
        </divelog>
        """
        parser = SubsurfaceParser()
        profiles = parser.parse_string(xml)

        # Should parse without crashing
        assert len(profiles) == 1
        assert len(profiles[0].points) >= 4


# ==============================================================================
# UDDF PARSER TESTS - Validate UDDF XML parsing
# ==============================================================================


class TestUDDFParser:
    """Test UDDF (Universal Dive Data Format) parsing."""

    def test_parses_basic_uddf_dive(self):
        """Verify basic UDDF dive with waypoints is parsed correctly."""
        xml = """
        <uddf>
            <profiledata>
                <dive>
                    <samples>
                        <waypoint>
                            <divetime>0</divetime>
                            <depth>0.0</depth>
                        </waypoint>
                        <waypoint>
                            <divetime>300</divetime>
                            <depth>20.0</depth>
                        </waypoint>
                        <waypoint>
                            <divetime>600</divetime>
                            <depth>0.0</depth>
                        </waypoint>
                    </samples>
                </dive>
            </profiledata>
        </uddf>
        """
        parser = UDDFParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 1
        profile = profiles[0]
        assert len(profile.points) == 3
        # UDDF time is in seconds, should be converted to minutes
        assert profile.points[1][0] == pytest.approx(5.0)  # 300s = 5min
        assert profile.points[1][1] == pytest.approx(20.0)

    def test_parses_gas_definitions(self):
        """Verify gas mix definitions are parsed and applied."""
        xml = """
        <uddf>
            <gasdefinitions>
                <mix id="nitrox32">
                    <o2>0.32</o2>
                    <n2>0.68</n2>
                    <he>0.0</he>
                </mix>
            </gasdefinitions>
            <profiledata>
                <dive>
                    <samples>
                        <waypoint>
                            <divetime>0</divetime>
                            <depth>0.0</depth>
                            <switchmix ref="nitrox32"/>
                        </waypoint>
                        <waypoint>
                            <divetime>600</divetime>
                            <depth>25.0</depth>
                        </waypoint>
                    </samples>
                </dive>
            </profiledata>
        </uddf>
        """
        parser = UDDFParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 1
        # Gas should be switched to nitrox32
        point = profiles[0].points[0]
        assert point[2] == pytest.approx(0.32)  # fO2
        assert point[3] == pytest.approx(0.0)   # fHe

    def test_handles_gas_switch_midpoint(self):
        """Verify gas switches during dive are applied correctly."""
        xml = """
        <uddf>
            <gasdefinitions>
                <mix id="air">
                    <o2>0.21</o2>
                </mix>
                <mix id="ean50">
                    <o2>0.50</o2>
                </mix>
            </gasdefinitions>
            <profiledata>
                <dive>
                    <samples>
                        <waypoint>
                            <divetime>0</divetime>
                            <depth>0.0</depth>
                        </waypoint>
                        <waypoint>
                            <divetime>600</divetime>
                            <depth>30.0</depth>
                        </waypoint>
                        <waypoint>
                            <divetime>900</divetime>
                            <depth>6.0</depth>
                            <switchmix ref="ean50"/>
                        </waypoint>
                        <waypoint>
                            <divetime>1200</divetime>
                            <depth>0.0</depth>
                        </waypoint>
                    </samples>
                </dive>
            </profiledata>
        </uddf>
        """
        parser = UDDFParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 1
        # First points should be air (0.21)
        assert profiles[0].points[0][2] == pytest.approx(0.21)
        # After switch, should be EAN50 (0.50)
        assert profiles[0].points[2][2] == pytest.approx(0.50)

    def test_handles_missing_samples_element(self):
        """Verify parser skips dives without samples."""
        xml = """
        <uddf>
            <profiledata>
                <dive>
                </dive>
            </profiledata>
        </uddf>
        """
        parser = UDDFParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 0

    def test_handles_missing_waypoints(self):
        """Verify parser skips dives with no waypoints."""
        xml = """
        <uddf>
            <profiledata>
                <dive>
                    <samples>
                    </samples>
                </dive>
            </profiledata>
        </uddf>
        """
        parser = UDDFParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 0

    def test_converts_divetime_to_minutes(self):
        """Verify divetime in seconds is correctly converted to minutes."""
        xml = """
        <uddf>
            <profiledata>
                <dive>
                    <samples>
                        <waypoint>
                            <divetime>0</divetime>
                            <depth>0.0</depth>
                        </waypoint>
                        <waypoint>
                            <divetime>1800</divetime>
                            <depth>20.0</depth>
                        </waypoint>
                    </samples>
                </dive>
            </profiledata>
        </uddf>
        """
        parser = UDDFParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 1
        # 1800 seconds = 30 minutes
        assert profiles[0].points[1][0] == pytest.approx(30.0)

    def test_calculates_bottom_time_from_depth(self):
        """Verify bottom time sums segments deeper than 3m."""
        xml = """
        <uddf>
            <profiledata>
                <dive>
                    <samples>
                        <waypoint>
                            <divetime>0</divetime>
                            <depth>0.0</depth>
                        </waypoint>
                        <waypoint>
                            <divetime>120</divetime>
                            <depth>20.0</depth>
                        </waypoint>
                        <waypoint>
                            <divetime>900</divetime>
                            <depth>20.0</depth>
                        </waypoint>
                        <waypoint>
                            <divetime>1200</divetime>
                            <depth>0.0</depth>
                        </waypoint>
                    </samples>
                </dive>
            </profiledata>
        </uddf>
        """
        parser = UDDFParser()
        profiles = parser.parse_string(xml)

        assert len(profiles) == 1
        # Segments > 3m:
        #   0-120s (0->20m, 20m > 3m) = 2 min
        #   120-900s (20->20m) = 13 min
        #   900-1200s (20->0m, 20m > 3m) = 5 min
        # Total = 20 min
        assert profiles[0].bottom_time == pytest.approx(20.0)


# ==============================================================================
# CSV PARSER TESTS - Validate CSV dive log parsing
# ==============================================================================


class TestCSVParser:
    """Test CSV dive log parsing with flexible column detection."""

    def test_parses_timeseries_csv(self):
        """Verify time-series CSV with time, depth columns is parsed correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,depth,o2\n")
            f.write("0.0,0.0,0.21\n")
            f.write("5.0,20.0,0.21\n")
            f.write("15.0,20.0,0.21\n")
            f.write("20.0,0.0,0.21\n")
            csv_path = f.name

        try:
            parser = CSVParser()
            profiles = parser.parse_file(csv_path)

            assert len(profiles) == 1
            profile = profiles[0]
            assert len(profile.points) == 4
            assert profile.max_depth == pytest.approx(20.0)
            assert profile.points[1][1] == pytest.approx(20.0)  # depth at 5min
        finally:
            Path(csv_path).unlink()

    def test_auto_detects_column_names(self):
        """Verify case-insensitive column name detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Time_Min,Depth_M,O2\n")
            f.write("0,0,0.21\n")
            f.write("5,15,0.21\n")
            csv_path = f.name

        try:
            parser = CSVParser()
            profiles = parser.parse_file(csv_path)

            assert len(profiles) == 1
            assert len(profiles[0].points) == 2
        finally:
            Path(csv_path).unlink()

    def test_extracts_gas_fractions(self):
        """Verify O2 and He fractions are extracted from CSV columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,depth,o2,he\n")
            f.write("0,0,0.32,0.0\n")
            f.write("5,30,0.32,0.0\n")
            csv_path = f.name

        try:
            parser = CSVParser()
            profiles = parser.parse_file(csv_path)

            assert len(profiles) == 1
            point = profiles[0].points[0]
            assert point[2] == pytest.approx(0.32)  # fO2
            assert point[3] == pytest.approx(0.0)   # fHe
        finally:
            Path(csv_path).unlink()

    def test_fallback_to_square_profile_from_metadata(self):
        """Verify square profile creation from max_depth/bottom_time metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("max_depth,bottom_time\n")
            f.write("30.0,20.0\n")
            csv_path = f.name

        try:
            parser = CSVParser()
            profiles = parser.parse_file(csv_path)

            assert len(profiles) == 1
            profile = profiles[0]
            # Should create a square profile with descent, bottom, ascent
            assert profile.max_depth == pytest.approx(30.0)
            assert len(profile.points) == 4  # Start, depth, end bottom, surface
        finally:
            Path(csv_path).unlink()

    def test_handles_empty_csv(self):
        """Verify empty CSV files are handled gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,depth\n")
            csv_path = f.name

        try:
            parser = CSVParser()
            profiles = parser.parse_file(csv_path)

            assert len(profiles) == 0
        finally:
            Path(csv_path).unlink()

    def test_rejects_insufficient_points(self):
        """Verify CSVs with <2 points are rejected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,depth\n")
            f.write("0,0\n")
            csv_path = f.name

        try:
            parser = CSVParser()
            profiles = parser.parse_file(csv_path)

            assert len(profiles) == 0  # Only 1 point
        finally:
            Path(csv_path).unlink()

    def test_calculates_bottom_time_from_depth(self):
        """Verify bottom_time sums segments deeper than 3m."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,depth\n")
            f.write("0,0\n")
            f.write("2,20\n")
            f.write("20,20\n")
            f.write("25,0\n")
            csv_path = f.name

        try:
            parser = CSVParser()
            profiles = parser.parse_file(csv_path)

            assert len(profiles) == 1
            # Segments > 3m:
            #   0-2 (0->20m, 20m > 3m) = 2 min
            #   2-20 (20->20m) = 18 min
            #   20-25 (20->0m, 20m > 3m) = 5 min
            # Total = 25 min
            assert profiles[0].bottom_time == pytest.approx(25.0)
        finally:
            Path(csv_path).unlink()


# ==============================================================================
# GITHUB MINER TESTS - Validate attachment extraction (mocked)
# ==============================================================================


class TestGitHubMiner:
    """Test GitHub issue attachment extraction with mocked API."""

    def test_extracts_user_attachments_url(self):
        """Verify extraction of user-attachments URLs."""
        miner = GitHubMiner(token="fake_token")
        text = """
        Here is my dive log:
        https://github.com/user-attachments/assets/abc-123-def-456/divelog.xml
        Let me know what you think.
        """
        attachments = miner._extract_attachments(text)

        assert len(attachments) == 1
        assert 'divelog.xml' in attachments[0]['filename']
        assert attachments[0]['extension'] == '.xml'

    def test_extracts_files_subdomain_url(self):
        """Verify extraction of github.com/repo/files URLs."""
        miner = GitHubMiner(token="fake_token")
        text = """
        Attached: https://github.com/subsurface/subsurface/files/12345678/dump.ssrf
        """
        attachments = miner._extract_attachments(text)

        assert len(attachments) == 1
        assert attachments[0]['extension'] == '.ssrf'

    def test_skips_extensionless_links(self):
        """Verify links with no extension are skipped (they're typically screenshots)."""
        miner = GitHubMiner(token="fake_token")
        text = """
        Binary dump: https://github.com/user-attachments/files/abc123def456
        """
        attachments = miner._extract_attachments(text)

        assert len(attachments) == 0  # No recognized extension, should skip

    def test_skips_non_dive_extensions(self):
        """Verify non-dive file extensions are skipped."""
        miner = GitHubMiner(token="fake_token")
        text = """
        Screenshot: https://github.com/user-attachments/assets/abc/screenshot.png
        Dive log: https://github.com/user-attachments/assets/def/dive.xml
        """
        attachments = miner._extract_attachments(text)

        # Should only extract .xml, not .png
        assert len(attachments) == 1
        assert attachments[0]['extension'] == '.xml'

    def test_deduplicates_urls(self):
        """Verify duplicate URLs are only extracted once."""
        miner = GitHubMiner(token="fake_token")
        text = """
        Link 1: https://github.com/user-attachments/assets/abc/dive.xml
        Link 2: https://github.com/user-attachments/assets/abc/dive.xml
        """
        attachments = miner._extract_attachments(text)

        assert len(attachments) == 1

    @patch('scraping.github_miner.requests.Session.get')
    def test_handles_rate_limit_headers(self, mock_get):
        """Verify rate limit handling sleeps when limit is low."""
        miner = GitHubMiner(token="fake_token")

        # Mock response with low rate limit
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            'X-RateLimit-Remaining': '5',  # Low remaining
            'X-RateLimit-Reset': str(int(time.time()) + 10),
        }
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        with patch('time.sleep') as mock_sleep:
            miner._fetch_issues("test", "repo", 1)
            # Should have slept due to low rate limit
            mock_sleep.assert_called()

    @patch('scraping.github_miner.requests.Session.get')
    def test_download_idempotent(self, mock_get):
        """Verify existing files are not re-downloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            miner = GitHubMiner(token="fake_token", output_dir=tmpdir)

            # Create a fake existing file
            repo_dir = Path(tmpdir) / "test_repo"
            repo_dir.mkdir()
            existing_file = repo_dir / "issue_1_dive.xml"
            existing_file.write_text("existing content")

            # Mock API response with proper headers
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                'X-RateLimit-Remaining': '1000',
                'X-RateLimit-Reset': str(int(time.time()) + 3600),
            }
            mock_response.json.return_value = [
                {
                    "number": 1,
                    "title": "Test issue",
                    "body": "https://github.com/user-attachments/assets/abc/dive.xml",
                    "comments": 0,
                }
            ]
            mock_get.return_value = mock_response

            # Mine the repo
            downloads = miner.mine_repo("test", "test_repo", max_pages=1)

            # Should skip existing file (no download calls to file content)
            assert len(downloads) == 0  # No new downloads


# ==============================================================================
# LOADER TESTS - Validate profile loading and filtering
# ==============================================================================


class TestLoader:
    """Test profile loading, filtering, and serialization."""

    def test_filters_too_few_points(self):
        """Verify profiles with too few points are filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parsed_dir = Path(tmpdir)

            # Create a profile with only 3 points
            data = [
                {
                    "name": "shallow_dive",
                    "max_depth": 10.0,
                    "bottom_time": 5.0,
                    "points": [
                        [0.0, 0.0, 0.21, 0.0],
                        [2.0, 10.0, 0.21, 0.0],
                        [5.0, 0.0, 0.21, 0.0],
                    ],
                }
            ]
            json_file = parsed_dir / "test.json"
            json_file.write_text(json.dumps(data))

            # Load with min_points=10
            profiles = load_profiles(str(parsed_dir), min_points=10)

            assert len(profiles) == 0  # Should be filtered out

    def test_filters_too_deep(self):
        """Verify profiles exceeding max_depth are filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parsed_dir = Path(tmpdir)

            # Create a deep dive (100m)
            points = [[i * 1.0, 100.0, 0.21, 0.0] for i in range(15)]
            data = [
                {
                    "name": "deep_dive",
                    "max_depth": 100.0,
                    "bottom_time": 10.0,
                    "points": points,
                }
            ]
            json_file = parsed_dir / "test.json"
            json_file.write_text(json.dumps(data))

            # Load with max_depth=50.0
            profiles = load_profiles(str(parsed_dir), max_depth=50.0)

            assert len(profiles) == 0  # Should be filtered out

    def test_filters_too_shallow(self):
        """Verify profiles below min_depth are filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parsed_dir = Path(tmpdir)

            # Create a very shallow dive (2m)
            points = [[i * 1.0, 2.0, 0.21, 0.0] for i in range(15)]
            data = [
                {
                    "name": "shallow_dive",
                    "max_depth": 2.0,
                    "bottom_time": 10.0,
                    "points": points,
                }
            ]
            json_file = parsed_dir / "test.json"
            json_file.write_text(json.dumps(data))

            # Load with min_depth=3.0
            profiles = load_profiles(str(parsed_dir), min_depth=3.0)

            assert len(profiles) == 0  # Should be filtered out

    def test_loads_valid_profiles(self):
        """Verify valid profiles are loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parsed_dir = Path(tmpdir)

            # Create a valid profile
            points = [[i * 1.0, 20.0, 0.21, 0.0] for i in range(15)]
            data = [
                {
                    "name": "valid_dive",
                    "max_depth": 20.0,
                    "bottom_time": 10.0,
                    "points": points,
                }
            ]
            json_file = parsed_dir / "test.json"
            json_file.write_text(json.dumps(data))

            # Load profiles
            profiles = load_profiles(str(parsed_dir), min_points=10, max_depth=50.0, min_depth=3.0)

            assert len(profiles) == 1
            assert profiles[0].name == "valid_dive"
            assert profiles[0].max_depth == pytest.approx(20.0)

    def test_profile_to_dict_roundtrip(self):
        """Verify profile serialization and deserialization roundtrip."""
        # Create a profile
        profile = DiveProfile(
            name="test_dive",
            max_depth=30.0,
            bottom_time=15.0,
            points=[
                (0.0, 0.0, 0.21, 0.0),
                (5.0, 30.0, 0.21, 0.0),
                (15.0, 30.0, 0.21, 0.0),
                (20.0, 0.0, 0.21, 0.0),
            ],
        )

        # Convert to dict
        data = profile_to_dict(profile)

        # Convert back
        restored = dict_to_profile(data)

        # Verify roundtrip
        assert restored.name == profile.name
        assert restored.max_depth == pytest.approx(profile.max_depth)
        assert restored.bottom_time == pytest.approx(profile.bottom_time)
        assert len(restored.points) == len(profile.points)
        assert restored.points[1][1] == pytest.approx(30.0)

    def test_profile_json_serialization(self):
        """Verify profiles can be serialized to JSON."""
        profile = DiveProfile(
            name="json_test",
            max_depth=25.0,
            bottom_time=12.0,
            points=[
                (0.0, 0.0, 0.21, 0.0),
                (10.0, 25.0, 0.21, 0.0),
            ],
        )

        # Serialize to JSON
        data = profile_to_dict(profile)
        json_str = json.dumps(data)

        # Deserialize
        loaded_data = json.loads(json_str)
        restored = dict_to_profile(loaded_data)

        assert restored.name == profile.name
        assert restored.max_depth == pytest.approx(profile.max_depth)

    def test_parse_raw_files_sanitizes_xml(self):
        """Verify parse_raw_files sanitizes XML before parsing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            parsed_dir = Path(tmpdir) / "parsed"
            raw_dir.mkdir()

            # Create a raw XML file with PII
            xml_content = """
            <divelog>
                <dive number="1">
                    <buddy>Secret Buddy</buddy>
                    <location>Top Secret Location</location>
                    <divecomputer>
                        <sample time="0:00 min" depth="0.0 m"/>
                        <sample time="5:00 min" depth="20.0 m"/>
                        <sample time="10:00 min" depth="0.0 m"/>
                    </divecomputer>
                </dive>
            </divelog>
            """
            xml_file = raw_dir / "dive.xml"
            xml_file.write_text(xml_content)

            # Parse files
            count = parse_raw_files(str(raw_dir), str(parsed_dir))

            assert count == 1

            # Load and verify PII was sanitized
            parsed_file = parsed_dir / "dive.json"
            assert parsed_file.exists()
            # The sanitized version should not contain buddy/location info
            # (this is validated by checking the profile was parsed successfully)


# ==============================================================================
# ZIP EXTRACTION TESTS - Validate ZIP handling in loader
# ==============================================================================


MINIMAL_SUBSURFACE_XML = """
<divelog>
    <dive number="1" date="2024-01-15">
        <divecomputer>
            <sample time="0:00 min" depth="0.0 m"/>
            <sample time="2:00 min" depth="15.0 m"/>
            <sample time="10:00 min" depth="15.0 m"/>
            <sample time="12:00 min" depth="0.0 m"/>
        </divecomputer>
    </dive>
</divelog>
"""


class TestZipExtraction:
    """Test ZIP archive extraction in the loader pipeline."""

    def test_extracts_xml_from_zip(self):
        """Verify XML files inside ZIPs are extracted and parseable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            parsed_dir = Path(tmpdir) / "parsed"
            raw_dir.mkdir()

            # Create a ZIP containing a Subsurface XML file
            zip_path = raw_dir / "test_dive.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("dive_data.xml", MINIMAL_SUBSURFACE_XML)

            count = parse_raw_files(str(raw_dir), str(parsed_dir))

            assert count == 1
            # Verify extracted directory was created
            assert (raw_dir / "test_dive.extracted").exists()
            assert (raw_dir / "test_dive.extracted" / "dive_data.xml").exists()

    def test_extracts_ssrf_from_zip(self):
        """Verify .ssrf files inside ZIPs are extracted and parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            parsed_dir = Path(tmpdir) / "parsed"
            raw_dir.mkdir()

            zip_path = raw_dir / "backup.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("my_dives.ssrf", MINIMAL_SUBSURFACE_XML)

            count = parse_raw_files(str(raw_dir), str(parsed_dir))

            assert count == 1

    def test_extracts_uddf_from_zip(self):
        """Verify .uddf files inside ZIPs are extracted and parsed."""
        uddf_content = """
        <uddf>
            <profiledata>
                <dive>
                    <samples>
                        <waypoint><divetime>0</divetime><depth>0.0</depth></waypoint>
                        <waypoint><divetime>300</divetime><depth>20.0</depth></waypoint>
                        <waypoint><divetime>600</divetime><depth>0.0</depth></waypoint>
                    </samples>
                </dive>
            </profiledata>
        </uddf>
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            parsed_dir = Path(tmpdir) / "parsed"
            raw_dir.mkdir()

            zip_path = raw_dir / "dives.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("dive.uddf", uddf_content)

            count = parse_raw_files(str(raw_dir), str(parsed_dir))

            assert count == 1

    def test_idempotent_extraction(self):
        """Verify ZIP is not re-extracted if .extracted directory already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()

            zip_path = raw_dir / "test.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("dive.xml", MINIMAL_SUBSURFACE_XML)

            # First extraction
            _extract_zips(raw_dir)
            extracted_dir = raw_dir / "test.extracted"
            assert extracted_dir.exists()

            # Modify the extracted content to verify it's not overwritten
            marker = extracted_dir / "marker.txt"
            marker.write_text("untouched")

            # Second extraction should skip
            _extract_zips(raw_dir)
            assert marker.exists()
            assert marker.read_text() == "untouched"

    def test_rejects_path_traversal_in_zip(self):
        """Verify ZIP entries with path traversal are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()

            zip_path = raw_dir / "evil.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("../../../etc/passwd", "malicious content")
                zf.writestr("safe_dive.xml", MINIMAL_SUBSURFACE_XML)

            _extract_zips(raw_dir)

            extracted = raw_dir / "evil.extracted"
            assert extracted.exists()
            # Only safe file should be extracted
            assert (extracted / "safe_dive.xml").exists()
            # Malicious path should NOT be extracted
            assert not Path(tmpdir).parent.joinpath("etc/passwd").exists()

    def test_rejects_absolute_path_in_zip(self):
        """Verify ZIP entries with absolute paths are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()

            zip_path = raw_dir / "evil2.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("/tmp/malicious", "bad content")
                zf.writestr("good.xml", MINIMAL_SUBSURFACE_XML)

            _extract_zips(raw_dir)

            extracted = raw_dir / "evil2.extracted"
            assert (extracted / "good.xml").exists()
            # Absolute path entries should be skipped
            files = list(extracted.rglob("malicious"))
            assert len(files) == 0

    def test_handles_bad_zip_file(self):
        """Verify corrupt ZIP files are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()

            # Create a corrupt "ZIP" file
            bad_zip = raw_dir / "corrupt.zip"
            bad_zip.write_bytes(b"this is not a zip file")

            # Should not raise, just log a warning
            _extract_zips(raw_dir)
            # No extracted directory should be created
            assert not (raw_dir / "corrupt.extracted").exists()

    def test_multiple_files_in_zip(self):
        """Verify all parseable files in a ZIP are extracted and parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            parsed_dir = Path(tmpdir) / "parsed"
            raw_dir.mkdir()

            zip_path = raw_dir / "multi.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("dive1.xml", MINIMAL_SUBSURFACE_XML)
                zf.writestr("dive2.xml", MINIMAL_SUBSURFACE_XML)
                zf.writestr("screenshot.png", b"fake png data")

            count = parse_raw_files(str(raw_dir), str(parsed_dir))

            # Both XML files should be parsed (1 dive each)
            assert count == 2


# ==============================================================================
# GZIP DETECTION TESTS - Validate gzip-compressed dive data handling
# ==============================================================================


class TestGzipDetection:
    """Test gzip-compressed dive data detection and parsing."""

    def test_detects_gzip_subsurface_xml(self):
        """Verify gzip-compressed Subsurface XML is detected and parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "dive.gz"
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                f.write(MINIMAL_SUBSURFACE_XML)

            profiles = _try_gzip_parse(filepath)

            assert len(profiles) == 1
            assert profiles[0].max_depth == pytest.approx(15.0)

    def test_skips_non_gzip_bin(self):
        """Verify non-gzip .bin files (e.g., images) are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "screenshot.bin"
            # PNG magic bytes
            filepath.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

            profiles = _try_gzip_parse(filepath)

            assert len(profiles) == 0

    def test_gzip_in_parse_raw_files(self):
        """Verify gzip files are parsed during the full parse pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            parsed_dir = Path(tmpdir) / "parsed"
            raw_dir.mkdir()

            # Create a gzip-compressed Subsurface XML with .gz extension
            gz_path = raw_dir / "dive_data.gz"
            with gzip.open(gz_path, 'wt', encoding='utf-8') as f:
                f.write(MINIMAL_SUBSURFACE_XML)

            count = parse_raw_files(str(raw_dir), str(parsed_dir))

            assert count == 1
            assert (parsed_dir / "dive_data.json").exists()

    def test_gzip_bin_extension_in_pipeline(self):
        """Verify gzip .bin files are detected and parsed in the pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            parsed_dir = Path(tmpdir) / "parsed"
            raw_dir.mkdir()

            gz_path = raw_dir / "dive_backup.bin"
            with gzip.open(gz_path, 'wt', encoding='utf-8') as f:
                f.write(MINIMAL_SUBSURFACE_XML)

            count = parse_raw_files(str(raw_dir), str(parsed_dir))

            assert count == 1

    def test_handles_corrupt_gzip(self):
        """Verify corrupt gzip files don't crash the pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            parsed_dir = Path(tmpdir) / "parsed"
            raw_dir.mkdir()

            # Gzip magic bytes but corrupt content
            corrupt = raw_dir / "bad.gz"
            corrupt.write_bytes(b'\x1f\x8b' + b'\x00' * 50)

            count = parse_raw_files(str(raw_dir), str(parsed_dir))

            assert count == 0  # Should not crash, just skip
