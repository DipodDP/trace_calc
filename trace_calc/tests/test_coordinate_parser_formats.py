"""Tests for CoordinateParser format compatibility."""

import pytest
import re
from trace_calc.application.services.user_input_parser import CoordinateParser
from trace_calc.domain.models.coordinates import Coordinates


class TestCoordinateParserFormats:
    """Test suite for coordinate format support."""

    def setup_method(self):
        self.parser = CoordinateParser()

    def test_decimal_degrees_with_commas(self):
        """Test: 55,3672698 91,646198 55,9896421 92,8998994"""
        coords = self.parser.parse("55,3672698 91,646198 55,9896421 92,8998994")
        assert len(coords) == 2
        assert coords[0].lat == pytest.approx(55.3672698, rel=1e-6)
        assert coords[0].lon == pytest.approx(91.646198, rel=1e-6)
        assert coords[1].lat == pytest.approx(55.9896421, rel=1e-6)
        assert coords[1].lon == pytest.approx(92.8998994, rel=1e-6)

    def test_decimal_degrees_with_semicolons(self):
        """Test: 55,3672698 91,646198; 55,9896421 92,8998994"""
        coords = self.parser.parse("55,3672698 91,646198; 55,9896421 92,8998994")
        assert len(coords) == 2
        assert coords[0].lat == pytest.approx(55.3672698, rel=1e-6)
        assert coords[1].lon == pytest.approx(92.8998994, rel=1e-6)

    def test_dms_format_russian_cardinals(self):
        """Test: 55°59'37.13"С  92°54'5.54"В  55°23'0.53"С 91°37'41.09"В"""
        coords = self.parser.parse('55°59\'37.13"С  92°54\'5.54"В  55°23\'0.53"С 91°37\'41.09"В')
        assert len(coords) == 2
        # 55°59'37.13" = 55 + 59/60 + 37.13/3600 ≈ 55.9936472
        assert coords[0].lat == pytest.approx(55.9936472, rel=1e-5)
        # 92°54'5.54" = 92 + 54/60 + 5.54/3600 ≈ 92.9015389
        assert coords[0].lon == pytest.approx(92.9015389, rel=1e-5)

    def test_dms_format_with_west_cardinal(self):
        """Test: 64° 44' 20.37"С  177° 29' 11.64"В  66° 18' 47.69"С  179° 8' 49.12"З"""
        coords = self.parser.parse('64° 44\' 20.37"С  177° 29\' 11.64"В  66° 18\' 47.69"С  179° 8\' 49.12"З')
        assert len(coords) == 2
        # 64°44'20.37" ≈ 64.7389917
        assert coords[0].lat == pytest.approx(64.7389917, rel=1e-5)
        # 179°8'49.12"W should be negative: -(179 + 8/60 + 49.12/3600) ≈ -179.1469778
        assert coords[1].lon == pytest.approx(-179.1469778, rel=1e-5)

    def test_space_separated_dms(self):
        """Test: 55 59 37.13,  92 54 5.54,  55 23 0.53, 91 37 41.09"""
        coords = self.parser.parse("55 59 37.13,  92 54 5.54,  55 23 0.53, 91 37 41.09")
        assert len(coords) == 2
        assert coords[0].lat == pytest.approx(55.9936472, rel=1e-5)
        assert coords[1].lon == pytest.approx(91.6280806, rel=1e-5)

    def test_comma_separated_dms_components(self):
        """Test: 55, 59, 37,13,  92, 54, 5,54,  55, 23, 0.53, 91, 37, 41,09"""
        coords = self.parser.parse("55, 59, 37,13,  92, 54, 5,54,  55, 23, 0.53, 91, 37, 41,09")
        assert len(coords) == 2
        assert coords[0].lat == pytest.approx(55.9936472, rel=1e-5)
        assert coords[1].lon == pytest.approx(91.6280806, rel=1e-5)

    def test_parse_with_names_basic(self):
        """Test parse_with_names with basic input."""
        site_names, coords_dec, coords_fmt = self.parser.parse_with_names(
            "Site A Site B",
            "55.7558 37.6173 59.9343 30.3351"
        )
        assert site_names == ["Site", "A"]
        assert coords_dec[1] == pytest.approx(37.6173, rel=1e-5)
        assert len(coords_fmt) == 4
        assert "N" in coords_fmt[0]
        assert "E" in coords_fmt[1]

    def test_parse_with_names_comma_separator(self):
        """Test parse_with_names with comma-separated names."""
        site_names, coords_dec, coords_fmt = self.parser.parse_with_names(
            "Moscow,Petersburg",
            "55.7558 37.6173 59.9343 30.3351"
        )
        assert site_names == ["Moscow", "Petersburg"]

    def test_parse_with_names_semicolon_separator(self):
        """Test parse_with_names with semicolon-separated names."""
        site_names, coords_dec, coords_fmt = self.parser.parse_with_names(
            "Point1;Point2",
            "55.7558 37.6173 59.9343 30.3351"
        )
        assert site_names == ["Point1", "Point2"]

    def test_parse_with_names_default_names(self):
        """Test parse_with_names with empty names (should use defaults)."""
        site_names, coords_dec, coords_fmt = self.parser.parse_with_names(
            "",
            "55.7558 37.6173 59.9343 30.3351"
        )
        assert site_names == ["Site A", "Site B"]

    def test_parse_with_names_single_name(self):
        """Test parse_with_names with only one name (should append default)."""
        site_names, coords_dec, coords_fmt = self.parser.parse_with_names(
            "Moscow",
            "55.7558 37.6173 59.9343 30.3351"
        )
        assert site_names == ["Moscow", "Site B"]

    def test_parse_with_names_russian_coordinates(self):
        """Test parse_with_names with DMS Russian format."""
        site_names, coords_dec, coords_fmt = self.parser.parse_with_names(
            "Москва Питер",
            '55°45\'25.4"С 37°37\'6.2"В 59°56\'10.4"С 30°20\'6.6"В'
        )
        assert site_names == ["Москва", "Питер"]
        assert len(coords_dec) == 4
        # 55°45'25.4" ≈ 55.757056
        assert coords_dec[0] == pytest.approx(55.757056, rel=1e-5)

    def test_parse_with_names_negative_coords_south(self):
        """Test parse_with_names with south/west (negative) coordinates."""
        site_names, coords_dec, coords_fmt = self.parser.parse_with_names(
            "A B",
            "10.5 S, 20.3 W, 15.2 N, 25.4 E"
        )
        assert coords_dec[0] == pytest.approx(-10.5, rel=1e-5)
        assert coords_dec[1] == pytest.approx(-20.3, rel=1e-5)
        assert coords_dec[2] == pytest.approx(15.2, rel=1e-5)
        assert coords_dec[3] == pytest.approx(25.4, rel=1e-5)
        assert "S" in coords_fmt[0]
        assert "W" in coords_fmt[1]

    def test_parse_with_names_formatting_output(self):
        """Test that formatted output contains proper hemisphere indicators."""
        site_names, coords_dec, coords_fmt = self.parser.parse_with_names(
            "A B",
            "55.7558 37.6173 -33.8688 151.2093"
        )
        assert "N" in coords_fmt[0]  # Positive latitude
        assert "E" in coords_fmt[1]  # Positive longitude
        assert "S" in coords_fmt[2]  # Negative latitude (Sydney)
        assert "E" in coords_fmt[3]  # Positive longitude

    def test_parse_with_names_invalid_coord_count(self):
        """Test parse_with_names raises error with wrong number of coordinates."""
        with pytest.raises(ValueError, match="Expected 2 coordinate pairs"):
            self.parser.parse_with_names("A B", "55.7558 37.6173")

    def test_parse_single_pair(self):
        """Test parsing a single coordinate pair."""
        coords = self.parser.parse("55.7558 37.6173")
        assert len(coords) == 1
        assert coords[0].lat == pytest.approx(55.7558, rel=1e-5)
        assert coords[0].lon == pytest.approx(37.6173, rel=1e-5)

    def test_parse_invalid_empty_string(self):
        """Test parsing empty string raises error."""
        with pytest.raises(ValueError, match="no values found"):
            self.parser.parse("")

    def test_parse_invalid_wrong_count(self):
        """Test parsing wrong number of values raises error."""
        with pytest.raises(ValueError, match="Expected 2 or 4 coordinate values"):
            self.parser.parse("55.7558 37.6173 59.9343")  # Only 3 values
