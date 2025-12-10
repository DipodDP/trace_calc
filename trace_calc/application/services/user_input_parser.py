import re
from trace_calc.domain.models.coordinates import Coordinates


class CoordinateParser:
    """
    Parses geographical coordinates in various formats.
    Supports decimal degrees and degrees-minutes-seconds (DMS) with different separators.
    """

    def __init__(self) -> None:
        self.hemisphere_multipliers = {
            "N": 1,
            "С": 1,
            "E": 1,
            "В": 1,
            "S": -1,
            "Ю": -1,
            "W": -1,
            "З": -1,
        }
        self.hemisphere_keys = list(self.hemisphere_multipliers.keys())

    def _is_number(self, s: str) -> bool:
        """Checks if a string can be converted to a float."""
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    def _dms_to_dd(self, d: float, m: float, s: float) -> float:
        """Converts degrees, minutes, seconds to decimal degrees."""
        sign = -1 if d < 0 else 1
        return sign * (abs(d) + m / 60.0 + s / 3600.0)

    def parse(self, text: str) -> list[Coordinates]:
        """
        Parses a string of coordinates and returns a list of one or two Coordinates objects.
        """
        text = text.strip()
        # Replace degree, minute, second symbols and semicolons with spaces
        text = re.sub(r"""[°'"]""", " ", text)
        text = re.sub(r";", " ", text)

        # Replace all commas with dots (assuming they are decimal separators after initial cleanup)
        text = text.replace(",", ".")

        text = re.sub(r"\s+", " ", text)
        # Strip any leading non-numeric characters
        text = re.sub(r"^[^0-9+-]+", "", text)

        # Tokenize the string into numbers (now only with '.' as decimal) and hemisphere characters
        tokens = re.findall(r"[+-]?\d+(?:\.\d+)?|[NSEWСВЮЗ]", text, re.IGNORECASE)

        if not tokens:
            raise ValueError("Invalid coordinate string: no values found.")

        values = []
        i = 0
        while i < len(tokens):
            is_dms = False
            if (
                i + 2 < len(tokens)
                and self._is_number(tokens[i])
                and self._is_number(tokens[i + 1])
                and self._is_number(tokens[i + 2])
            ):
                d_str, m_str, s_str = tokens[i], tokens[i + 1], tokens[i + 2]

                # If degrees or minutes part contains a decimal, it's not a DMS triplet.
                if "." in d_str or "." in m_str:
                    is_dms = False
                else:
                    d, m, s = float(d_str), float(m_str), float(s_str)
                    if 0 <= m < 60 and 0 <= s < 60:
                        is_dms = True

            if is_dms:
                d, m, s = float(tokens[i]), float(tokens[i + 1]), float(tokens[i + 2])
                val = self._dms_to_dd(d, m, s)
                i += 3
                if i < len(tokens) and tokens[i].upper() in self.hemisphere_keys:
                    if val >= 0:
                        val *= self.hemisphere_multipliers.get(tokens[i].upper(), 1)
                    i += 1
                values.append(val)
                continue

            # Fallback to parsing as a single decimal value
            if self._is_number(tokens[i]):
                val = float(tokens[i])
                i += 1
                if i < len(tokens) and tokens[i].upper() in self.hemisphere_keys:
                    if val >= 0:
                        val *= self.hemisphere_multipliers.get(tokens[i].upper(), 1)
                    i += 1
                values.append(val)
            else:
                # Not a number, just advance
                i += 1

        if len(values) not in [2, 4]:
            raise ValueError(
                f"Expected 2 or 4 coordinate values, but found {len(values)}."
            )

        if len(values) == 2:
            return [Coordinates(lat=values[0], lon=values[1])]
        return [
            Coordinates(lat=values[0], lon=values[1]),
            Coordinates(lat=values[2], lon=values[3]),
        ]

    def parse_with_names(
        self,
        s_name: str,
        coords: str,
        default_name_a: str = "Site A",
        default_name_b: str = "Site B",
    ) -> tuple[list[str], list[float], list[str]]:
        """
        Parse site names and coordinates together.

        Args:
            s_name: Site names separated by spaces, commas, or semicolons
            coords: Coordinate string in any supported format
            default_name_a: Default name for the first site
            default_name_b: Default name for the second site

        Returns:
            Tuple of (site_names_list, coords_decimal, coords_formatted):
            - site_names_list: List of 2 site names
            - coords_decimal: List of 4 floats [lat1, lon1, lat2, lon2]
            - coords_formatted: List of 4 formatted strings for display
        """
        # Parse site names
        s_name = s_name.replace(',', ' ').replace(';', ' ')
        site_names = [name for name in re.split(r'\s+', s_name) if name][:2]

        # Default site names if not provided
        if not site_names:
            site_names = [default_name_a, default_name_b]
        elif len(site_names) == 1:
            site_names.append(default_name_b)

        # Parse coordinates
        coordinates_list = self.parse(coords)

        if len(coordinates_list) != 2:
            raise ValueError(
                f"Expected 2 coordinate pairs, got {len(coordinates_list)}. "
                f"Please provide coordinates for both sites."
            )

        # Extract to decimal list format [lat1, lon1, lat2, lon2]
        coords_decimal = [
            coordinates_list[0].lat,
            coordinates_list[0].lon,
            coordinates_list[1].lat,
            coordinates_list[1].lon,
        ]

        # Format coordinates for display
        coords_formatted = []
        for i, coord_val in enumerate(coords_decimal):
            is_latitude = i % 2 == 0

            # Determine hemisphere
            if is_latitude:
                hemisphere = "N" if coord_val >= 0 else "S"
            else:
                hemisphere = "E" if coord_val >= 0 else "W"

            # Format as decimal degrees with hemisphere
            coords_formatted.append(f"{abs(coord_val):.6f}° {hemisphere}")

        return site_names, coords_decimal, coords_formatted
