import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.transformations.dynamic_zones.v1 import (
    calculate_least_squares_polygon,
    calculate_simplified_polygon,
)


def test_dynamic_zones_no_simplification_required():
    # given
    polygon = np.array([[10, 1], [10, 10], [20, 10], [20, 1]])

    # when
    simplified_polygon, _ = calculate_simplified_polygon(
        contours=[polygon],
        required_number_of_vertices=len(polygon),
    )

    # then
    assert np.allclose(
        simplified_polygon, np.array([[10, 1], [10, 10], [20, 10], [20, 1]])
    ), "Polygon should not be modified if it already contains required number of vertices"


def test_dynamic_zones_resulting_in_convex_polygon():
    # given
    polygon = np.array(
        [
            [10, 10],
            [10, 1],
            [15, 1],
            [15, 9],
            [16, 1],
            [20, 1],
            [20, 10],
            [18, 10],
            [17, 2],
            [16, 10],
        ]
    )

    # when
    simplified_polygon, _ = calculate_simplified_polygon(
        contours=[polygon],
        required_number_of_vertices=4,
    )

    # then
    assert np.allclose(
        simplified_polygon, np.array([[20, 10], [20, 1], [10, 1], [10, 10]])
    ), (
        "Valleys ([15, 1], [15, 9], [16, 1]) on the edge between [10, 1] and [20, 1] "
        "and ([18, 10], [17, 2], [16, 10]) on the edge between [20, 10] and [10, 10] "
        "should be dropped and shape of the polygon should remain unchanged"
    )


def test_dynamic_zones_drop_intermediate_points():
    # given
    polygon = np.array(
        np.array([[10, 10], [10, 5], [10, 1], [15, 1], [20, 1], [20, 5], [20, 10]])
    )

    # when
    simplified_polygon, _ = calculate_simplified_polygon(
        contours=[polygon],
        required_number_of_vertices=4,
    )

    # then
    assert np.allclose(
        simplified_polygon, np.array([[20, 10], [20, 1], [10, 1], [10, 10]])
    ), (
        "Intermediate points [10, 5] (between [10, 10] and [10, 1]), "
        "[15, 1] (between [10, 1] and [20, 1]) and [20, 5] (between [20, 1] and [20, 10]) "
        "should be dropped and shape of polygon should remain unchanged."
    )


def test_calculate_least_squares_polygon():
    # given
    polygon = np.array(
        np.array([[1631, 1550], [1575, 1737], [1682, 1768], [1701, 1556]])
    )
    contour = np.array(
        [
            [1631, 1550],
            [1629, 1552],
            [1629, 1553],
            [1626, 1556],
            [1626, 1557],
            [1623, 1560],
            [1623, 1561],
            [1620, 1564],
            [1620, 1565],
            [1617, 1568],
            [1617, 1569],
            [1614, 1572],
            [1614, 1573],
            [1612, 1575],
            [1612, 1637],
            [1611, 1638],
            [1611, 1639],
            [1609, 1641],
            [1609, 1642],
            [1606, 1645],
            [1606, 1646],
            [1604, 1648],
            [1604, 1649],
            [1603, 1650],
            [1598, 1650],
            [1595, 1653],
            [1595, 1654],
            [1593, 1656],
            [1593, 1662],
            [1592, 1663],
            [1592, 1664],
            [1590, 1666],
            [1590, 1667],
            [1587, 1670],
            [1587, 1671],
            [1585, 1673],
            [1585, 1674],
            [1584, 1675],
            [1579, 1675],
            [1578, 1676],
            [1578, 1677],
            [1576, 1679],
            [1576, 1680],
            [1575, 1681],
            [1575, 1737],
            [1576, 1738],
            [1576, 1739],
            [1578, 1741],
            [1578, 1742],
            [1579, 1743],
            [1584, 1743],
            [1585, 1744],
            [1585, 1745],
            [1587, 1747],
            [1587, 1748],
            [1590, 1751],
            [1590, 1752],
            [1592, 1754],
            [1592, 1755],
            [1593, 1756],
            [1593, 1762],
            [1595, 1764],
            [1595, 1765],
            [1598, 1768],
            [1682, 1768],
            [1685, 1765],
            [1685, 1764],
            [1687, 1762],
            [1692, 1762],
            [1693, 1761],
            [1693, 1760],
            [1695, 1758],
            [1695, 1757],
            [1696, 1756],
            [1696, 1750],
            [1697, 1749],
            [1697, 1748],
            [1700, 1745],
            [1700, 1744],
            [1701, 1743],
            [1701, 1731],
            [1702, 1730],
            [1702, 1729],
            [1704, 1727],
            [1704, 1726],
            [1707, 1723],
            [1707, 1722],
            [1709, 1720],
            [1709, 1719],
            [1710, 1718],
            [1710, 1700],
            [1709, 1699],
            [1709, 1698],
            [1707, 1696],
            [1707, 1695],
            [1704, 1692],
            [1704, 1691],
            [1702, 1689],
            [1702, 1688],
            [1701, 1687],
            [1701, 1556],
            [1698, 1553],
            [1698, 1552],
            [1696, 1550],
        ]
    )

    # when
    least_squares_polygon = calculate_least_squares_polygon(
        contour=contour, polygon=polygon
    )

    # then
    assert np.allclose(
        least_squares_polygon,
        np.array([[1632, 1550], [1561, 1742], [1682, 1778], [1759, 1556]]),
    ), "Correct least squares polygon should be calculated based on the contour and polygon."


def test_calculate_least_squares_polygon_with_midpoint_fraction():
    # given
    polygon = np.array(
        np.array([[1631, 1550], [1575, 1737], [1682, 1768], [1701, 1556]])
    )
    contour = np.array(
        [
            [1631, 1550],
            [1629, 1552],
            [1629, 1553],
            [1626, 1556],
            [1626, 1557],
            [1623, 1560],
            [1623, 1561],
            [1620, 1564],
            [1620, 1565],
            [1617, 1568],
            [1617, 1569],
            [1614, 1572],
            [1614, 1573],
            [1612, 1575],
            [1612, 1637],
            [1611, 1638],
            [1611, 1639],
            [1609, 1641],
            [1609, 1642],
            [1606, 1645],
            [1606, 1646],
            [1604, 1648],
            [1604, 1649],
            [1603, 1650],
            [1598, 1650],
            [1595, 1653],
            [1595, 1654],
            [1593, 1656],
            [1593, 1662],
            [1592, 1663],
            [1592, 1664],
            [1590, 1666],
            [1590, 1667],
            [1587, 1670],
            [1587, 1671],
            [1585, 1673],
            [1585, 1674],
            [1584, 1675],
            [1579, 1675],
            [1578, 1676],
            [1578, 1677],
            [1576, 1679],
            [1576, 1680],
            [1575, 1681],
            [1575, 1737],
            [1576, 1738],
            [1576, 1739],
            [1578, 1741],
            [1578, 1742],
            [1579, 1743],
            [1584, 1743],
            [1585, 1744],
            [1585, 1745],
            [1587, 1747],
            [1587, 1748],
            [1590, 1751],
            [1590, 1752],
            [1592, 1754],
            [1592, 1755],
            [1593, 1756],
            [1593, 1762],
            [1595, 1764],
            [1595, 1765],
            [1598, 1768],
            [1682, 1768],
            [1685, 1765],
            [1685, 1764],
            [1687, 1762],
            [1692, 1762],
            [1693, 1761],
            [1693, 1760],
            [1695, 1758],
            [1695, 1757],
            [1696, 1756],
            [1696, 1750],
            [1697, 1749],
            [1697, 1748],
            [1700, 1745],
            [1700, 1744],
            [1701, 1743],
            [1701, 1731],
            [1702, 1730],
            [1702, 1729],
            [1704, 1727],
            [1704, 1726],
            [1707, 1723],
            [1707, 1722],
            [1709, 1720],
            [1709, 1719],
            [1710, 1718],
            [1710, 1700],
            [1709, 1699],
            [1709, 1698],
            [1707, 1696],
            [1707, 1695],
            [1704, 1692],
            [1704, 1691],
            [1702, 1689],
            [1702, 1688],
            [1701, 1687],
            [1701, 1556],
            [1698, 1553],
            [1698, 1552],
            [1696, 1550],
        ]
    )

    # when
    least_squares_polygon = calculate_least_squares_polygon(
        contour=contour, polygon=polygon, midpoint_fraction=0.5
    )

    # then
    assert np.allclose(
        least_squares_polygon,
        np.array([[1639, 1550], [1567, 1726], [1668, 1837], [1761, 1556]]),
    ), "Correct least squares polygon should be calculated based on the contour and polygon."
