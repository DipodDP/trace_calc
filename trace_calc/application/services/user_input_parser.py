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
        Parses a string of coordinates and returns a list of Coordinates objects.
        The string can contain multiple coordinates on separate lines, with descriptive text.
        """
        all_coords = []

        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            line = re.sub(r"""[°'"]""", " ", line)
            line = line.replace(",", ".")

            tokens = re.findall(
                r"[+-]?\d+(?:\.\d+)?|[NSEWСВЮЗ]|[^\s;]+", line, re.IGNORECASE
            )

            i = 0
            while i < len(tokens):
                # Try to parse DMS: (d, m, s, [h])
                if (
                    i + 2 < len(tokens)
                    and self._is_number(tokens[i])
                    and self._is_number(tokens[i + 1])
                    and self._is_number(tokens[i + 2])
                ):
                    d_str, m_str, s_str = tokens[i], tokens[i + 1], tokens[i + 2]
                    if "." not in d_str and "." not in m_str:
                        d, m, s = float(d_str), float(m_str), float(s_str)
                        if not (0 <= m < 60 and 0 <= s < 60):
                            raise ValueError(
                                "Invalid DMS coordinate: minutes or seconds out of range"
                            )

                        val = self._dms_to_dd(d, m, s)
                        i += 3
                        if (
                            i < len(tokens)
                            and tokens[i].upper() in self.hemisphere_keys
                        ):
                            if val >= 0:
                                val *= self.hemisphere_multipliers.get(
                                    tokens[i].upper(), 1
                                )
                            i += 1
                        all_coords.append(val)
                        continue

                # Try to parse DD: (d, [h])
                if self._is_number(tokens[i]):
                    val = float(tokens[i])
                    i += 1
                    if i < len(tokens) and tokens[i].upper() in self.hemisphere_keys:
                        if val >= 0:
                            val *= self.hemisphere_multipliers.get(tokens[i].upper(), 1)
                        i += 1
                    all_coords.append(val)
                    continue

                i += 1

        if len(all_coords) % 2 != 0:
            raise ValueError(
                f"Found an odd number of coordinate values: {len(all_coords)}"
            )

        coords_list = []
        for i in range(0, len(all_coords), 2):
            coords_list.append(Coordinates(lat=all_coords[i], lon=all_coords[i + 1]))

        if not coords_list:
            if text.strip():
                raise ValueError("No coordinates found in the input text.")
            else:
                raise ValueError("Input cannot be empty.")

        return coords_list

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
        s_name = s_name.replace(",", " ").replace(";", " ")
        site_names = [name for name in re.split(r"\s+", s_name) if name][:2]

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
