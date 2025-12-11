import pytest
from trace_calc.application.services.user_input_parser import CoordinateParser
from trace_calc.domain.models.coordinates import Coordinates


@pytest.fixture
def parser():
    return CoordinateParser()


@pytest.mark.parametrize(
    "input_string, expected",
    [
        # Decimal degrees tests
        (
            "55.3672698 91.646198 55.9896421 92.8998994",
            [Coordinates(55.3672698, 91.646198), Coordinates(55.9896421, 92.8998994)],
        ),
        (
            "55,3672698 91,646198; 55,9896421 92,8998994",
            [Coordinates(55.3672698, 91.646198), Coordinates(55.9896421, 92.8998994)],
        ),
        ("-10.5 20.5 30.5 -40.5", [Coordinates(-10.5, 20.5), Coordinates(30.5, -40.5)]),
        # DMS tests
        (
            "55 59 37.13 N 92 54 5.54 E 55 23 0.53 N 91 37 41.09 E",
            [Coordinates(55.993647, 92.901538), Coordinates(55.38348, 91.62808)],
        ),
        (
            "55°59'37.13\"С 92°54'5.54\"В 55°23'0.53\"С 91°37'41.09\"В",
            [Coordinates(55.993647, 92.901538), Coordinates(55.38348, 91.62808)],
        ),
        (
            "64 44 20.37 С 177 29 11.64 В 66 18 47.69 С 179 8 49.12 З",
            [Coordinates(64.73899, 177.48656), Coordinates(66.31324, -179.14697)],
        ),
        # Mixed DMS and Decimal
        (
            "55.993647 92.901538 55 23 0.53 N 91 37 41.09 E",
            [Coordinates(55.993647, 92.901538), Coordinates(55.38348, 91.62808)],
        ),
        # Negative values
        (
            "-55 59 37.13 92 54 5.54 55.38348 -91.62808",
            [Coordinates(-55.993647, 92.901538), Coordinates(55.38348, -91.62808)],
        ),
        # Extra text and separators
        (
            "Site A: 55.36 N, 91.64 E; Site B: 55.98 N, 92.89 E",
            [Coordinates(55.36, 91.64), Coordinates(55.98, 92.89)],
        ),
        ("55 59 37N 92 54 05E", [Coordinates(55.993611, 92.901389)]),
    ],
)
def test_coordinate_parser_valid(parser, input_string, expected):
    result = parser.parse(input_string)
    assert len(result) == len(expected)
    for i in range(len(expected)):
        assert result[i].lat == pytest.approx(expected[i].lat, abs=1e-5)
        assert result[i].lon == pytest.approx(expected[i].lon, abs=1e-5)


@pytest.mark.parametrize(
    "input_string",
    [
        "",
        "10 20 30",  # not enough values
        "10 20 30 40 50",  # too many values
        "10 70 30 N 20 30 40 E 10 20 30 N 10 20 30 E",  # invalid minutes
        "nota valid string",
    ],
)
def test_coordinate_parser_invalid(parser, input_string):
    with pytest.raises(ValueError):
        parser.parse(input_string)


def test_parse_with_descriptive_text_and_multiple_points(parser):
    input_text = """
    Координаты точек: 

    предложенная точка:
    Широта: 63° 25' 38.80" С
    Долгота: 80° 32' 5.39" В

    Ярайнерское м.р. ДНС-1:
    Широта: 63.145278° N
    Долгота: 77.775833° E
    """
    coordinates = parser.parse(input_text)
    assert len(coordinates) == 2
    
    # First coordinate
    assert pytest.approx(coordinates[0].lat, abs=1e-6) == 63.42744444444444
    assert pytest.approx(coordinates[0].lon, abs=1e-6) == 80.53483055555555
    
    # Second coordinate
    assert pytest.approx(coordinates[1].lat, abs=1e-6) == 63.145278
    assert pytest.approx(coordinates[1].lon, abs=1e-6) == 77.775833
