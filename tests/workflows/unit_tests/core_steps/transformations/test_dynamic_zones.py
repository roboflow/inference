import numpy as np
import pytest
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


def test_dynamic_zones_tensor_native_stores_scaled_polygon_in_bboxes_metadata():
    # Mirrors the numpy behavior change from PR #2614: the per-box
    # POLYGON_KEY_IN_SV_DETECTIONS payload must carry the SCALED polygon
    # (assignment happens after scale_polygon), not the pre-scale one.
    pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    import torch

    from inference.core.workflows.core_steps.transformations.dynamic_zones.v1_tensor import (
        OUTPUT_KEY as TENSOR_OUTPUT_KEY,
    )
    from inference.core.workflows.core_steps.transformations.dynamic_zones.v1_tensor import (
        OUTPUT_KEY_DETECTIONS as TENSOR_OUTPUT_KEY_DETECTIONS,
    )
    from inference.core.workflows.core_steps.transformations.dynamic_zones.v1_tensor import (
        DynamicZonesBlockV1 as TensorDynamicZonesBlockV1,
    )
    from inference.core.workflows.execution_engine.constants import (
        POLYGON_KEY_IN_SV_DETECTIONS,
    )
    from inference.core.workflows.execution_engine.entities.base import Batch
    from inference_models.models.base.instance_segmentation import InstanceDetections

    # given - a single square instance and a scale ratio that visibly moves vertices
    mask = np.zeros((1, 100, 100), dtype=bool)
    mask[0, 20:60, 30:70] = True
    detections = InstanceDetections(
        xyxy=torch.tensor([[30.0, 20.0, 70.0, 60.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=torch.from_numpy(mask),
        image_metadata={"class_names": {0: "zone"}},
        bboxes_metadata=[{"detection_id": "det-1"}],
    )
    block = TensorDynamicZonesBlockV1()

    # when
    result = block.run(
        predictions=Batch(content=[detections], indices=[(0,)]),
        required_number_of_vertices=4,
        scale_ratio=0.5,
    )

    # then
    scaled_polygons = result[0][TENSOR_OUTPUT_KEY]
    output_detections = result[0][TENSOR_OUTPUT_KEY_DETECTIONS]
    assert len(scaled_polygons) == 1
    stored_polygon = output_detections.bboxes_metadata[0][POLYGON_KEY_IN_SV_DETECTIONS]
    assert stored_polygon.shape == (
        len(scaled_polygons[0]),
        2,
    ), "Per-box polygon payload must be the (V, 2) polygon itself - numpy's sv COLUMN wrapping does not apply to per-box bboxes_metadata"
    assert (
        stored_polygon.tolist() == scaled_polygons[0]
    ), "Per-box polygon payload must equal the scaled polygon emitted in the zones output"
    unscaled_square = {(30, 20), (69, 20), (69, 59), (30, 59)}
    assert (
        set(map(tuple, stored_polygon.tolist())) != unscaled_square
    ), "Payload must not be the pre-scale polygon (scale_ratio=0.5 moves vertices)"
    assert output_detections.bboxes_metadata[0]["detection_id"] == "det-1"


def test_dynamic_zones_tensor_native_serialized_polygon_payload_matches_numpy():
    # End-to-end parity: the same segmentation input through the numpy block +
    # numpy serializer and through the tensor sibling + tensor serializer must
    # produce identical prediction dicts - in particular the declared-polygon
    # response field must not be nested one extra level (a (1, V, 2) payload
    # bypasses the serializer's declared-polygon fast path and nests the field).
    pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    import torch

    from inference.core.workflows.core_steps.common.serializers import (
        serialise_sv_detections as numpy_serialise,
    )
    from inference.core.workflows.core_steps.common.serializers_tensor import (
        serialise_sv_detections as tensor_serialise,
    )
    from inference.core.workflows.core_steps.transformations.dynamic_zones.v1 import (
        OUTPUT_KEY_DETECTIONS,
        DynamicZonesBlockV1,
    )
    from inference.core.workflows.core_steps.transformations.dynamic_zones.v1_tensor import (
        OUTPUT_KEY_DETECTIONS as TENSOR_OUTPUT_KEY_DETECTIONS,
    )
    from inference.core.workflows.core_steps.transformations.dynamic_zones.v1_tensor import (
        DynamicZonesBlockV1 as TensorDynamicZonesBlockV1,
    )
    from inference.core.workflows.execution_engine.constants import (
        POLYGON_KEY_IN_SV_DETECTIONS,
    )
    from inference.core.workflows.execution_engine.entities.base import Batch
    from inference_models.models.base.instance_segmentation import InstanceDetections

    # given - identical single-instance input in both representations
    mask = np.zeros((1, 100, 100), dtype=bool)
    mask[0, 20:60, 30:70] = True
    sv_detections = sv.Detections(
        xyxy=np.array([[30.0, 20.0, 70.0, 60.0]], dtype=np.float32),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float32),
        mask=mask.copy(),
        data={
            "class_name": np.array(["zone"]),
            "detection_id": np.array(["det-1"]),
        },
    )
    native_detections = InstanceDetections(
        xyxy=torch.tensor([[30.0, 20.0, 70.0, 60.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=torch.from_numpy(mask.copy()),
        image_metadata={"class_names": {0: "zone"}},
        bboxes_metadata=[{"detection_id": "det-1"}],
    )

    # when
    numpy_result = DynamicZonesBlockV1().run(
        predictions=Batch(content=[sv_detections], indices=[(0,)]),
        required_number_of_vertices=4,
        scale_ratio=0.5,
    )
    tensor_result = TensorDynamicZonesBlockV1().run(
        predictions=Batch(content=[native_detections], indices=[(0,)]),
        required_number_of_vertices=4,
        scale_ratio=0.5,
    )

    # then
    numpy_serialised = numpy_serialise(numpy_result[0][OUTPUT_KEY_DETECTIONS])
    tensor_serialised = tensor_serialise(tensor_result[0][TENSOR_OUTPUT_KEY_DETECTIONS])
    numpy_prediction = numpy_serialised["predictions"][0]
    tensor_prediction = tensor_serialised["predictions"][0]
    assert (
        numpy_prediction[POLYGON_KEY_IN_SV_DETECTIONS]
        == tensor_prediction[POLYGON_KEY_IN_SV_DETECTIONS]
    ), "Declared-polygon response field must match numpy (no extra nesting level)"
    assert (
        numpy_serialised == tensor_serialised
    ), "Serialized dynamic_zones predictions must be identical across representations"
