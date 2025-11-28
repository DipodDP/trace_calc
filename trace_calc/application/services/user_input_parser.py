import re
from typing import List
from trace_calc.domain.models.coordinates import Coordinates


class CoordinateParser:
    """
    Parses a string containing geographical coordinates in various formats.
    It can handle decimal degrees and degrees-minutes-seconds (DMS) formats,
    with different separators.
    """

    def __init__(self):
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

    def parse(self, text: str) -> List[Coordinates]:
        """
        Parses a string of coordinates and returns a list of one or two Coordinates objects.
        """
        # Pre-process string: replace degree symbols, etc., with spaces
        text = text.strip()
        text = re.sub(r"""[°'"]""", " ", text)
        text = re.sub(r";", " ", text)
        text = re.sub(r"\s+", " ", text)
        # Strip any leading non-numeric characters
        text = re.sub(r"^[^0-9+-]+", "", text)

        # Tokenize the string into numbers and hemisphere characters
        tokens = re.findall(r"[+-]?\d+(?:[.,]\d+)?|[NSEWСВЮЗ]", text, re.IGNORECASE)

        if not tokens:
            raise ValueError("Invalid coordinate string: no values found.")

        values = []
        i = 0
        while i < len(tokens):
            # Replace comma with period for float conversion
            if "," in tokens[i]:
                tokens[i] = tokens[i].replace(",", ".")

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
