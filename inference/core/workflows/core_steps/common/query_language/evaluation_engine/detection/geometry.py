from typing import List, Tuple

import shapely


def is_point_in_zone(
    point: Tuple[int, int],
    zone: List[Tuple[float, float]],
) -> bool:
    point = shapely.geometry.Point(point[0], point[1])
    polygon = shapely.geometry.Polygon(
        [(zone_point[0], zone_point[1]) for zone_point in zone]
    )
    return point.within(polygon)
