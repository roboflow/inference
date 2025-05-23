from typing import List


def parse_comma_separated_values(values: str) -> List[str]:
    return [v.strip() for v in values.split(",")]
