from unittest import mock
from unittest.mock import MagicMock

from inference.core.active_learning import accounting
from inference.core.active_learning.accounting import (
    get_images_in_labeling_jobs_of_specific_batch,
    get_matching_labeling_batch,
    image_can_be_submitted_to_batch,
)


def test_get_matching_labeling_batch_when_matching_batch_exists() -> None:
    # given
    all_labeling_batches = [
        {
            "name": "Pip Package Upload",
            "numJobs": 0,
            "uploaded": {"_seconds": 1698060510, "_nanoseconds": 403000000},
            "images": 1,
            "id": "XXX",
        },
        {
            "name": "active-learning",
            "uploaded": {"_seconds": 1698060589, "_nanoseconds": 416000000},
            "numJobs": 1,
            "images": 2,
            "id": "YYY",
        },
    ]

    # when
    result = get_matching_labeling_batch(
        all_labeling_batches=all_labeling_batches, batch_name="active-learning"
    )

    # then
    assert result == all_labeling_batches[1]


def test_get_matching_labeling_batch_when_matching_batch_does_not_exist() -> None:
    # given
    all_labeling_batches = [
        {
            "name": "Pip Package Upload",
            "numJobs": 0,
            "uploaded": {"_seconds": 1698060510, "_nanoseconds": 403000000},
            "images": 1,
            "id": "XXX",
        },
    ]

    # when
    result = get_matching_labeling_batch(
        all_labeling_batches=all_labeling_batches, batch_name="active-learning"
    )

    # then
    assert result is None


def test_get_images_in_labeling_jobs_of_specific_batch_when_matching_jobs_found() -> (
    None
):
    # given
    all_labeling_jobs = [
        {
            "sourceBatch": "XXX/YYY",
            "created": {"_seconds": 1697044995, "_nanoseconds": 375000000},
            "numImages": 161,
            "status": "complete",
            "unannotated": 0,
            "annotated": 161,
        },
        {
            "sourceBatch": "XXX/ZZZ",
            "created": {"_seconds": 1697044995, "_nanoseconds": 375000000},
            "numImages": 192,
            "status": "complete",
            "unannotated": 0,
            "annotated": 161,
        },
        {
            "sourceBatch": "XXX/YYY",
            "created": {"_seconds": 1697044995, "_nanoseconds": 375000000},
            "numImages": 200,
            "status": "complete",
            "unannotated": 0,
            "annotated": 161,
        },
    ]

    # when
    result = get_images_in_labeling_jobs_of_specific_batch(
        all_labeling_jobs=all_labeling_jobs,
        batch_id="YYY",
    )

    # then
    assert result == 361


def test_get_images_in_labeling_jobs_of_specific_batch_when_no_matching_job_found() -> (
    None
):
    # given
    all_labeling_jobs = [
        {
            "sourceBatch": "XXX/ZZZ",
            "created": {"_seconds": 1697044995, "_nanoseconds": 375000000},
            "numImages": 192,
            "status": "complete",
            "unannotated": 0,
            "annotated": 161,
        },
    ]

    # when
    result = get_images_in_labeling_jobs_of_specific_batch(
        all_labeling_jobs=all_labeling_jobs,
        batch_id="YYY",
    )

    # then
    assert result == 0


def test_image_can_be_submitted_to_batch_when_max_images_not_set() -> None:
    # when
    result = image_can_be_submitted_to_batch(
        batch_name="some",
        workspace_id="workspace",
        dataset_id="project",
        max_batch_images=None,
        api_key="api-key",
    )

    # then
    assert result is True


@mock.patch.object(accounting, "get_roboflow_labeling_batches")
def test_image_can_be_submitted_to_batch_when_no_matching_batches(
    get_roboflow_labeling_batches_mock: MagicMock,
) -> None:
    # given
    get_roboflow_labeling_batches_mock.return_value = {
        "batches": [
            {
                "name": "Pip Package Upload",
                "numJobs": 0,
                "uploaded": {"_seconds": 1698060510, "_nanoseconds": 403000000},
                "images": 1,
                "id": "XXX",
            },
        ]
    }

    # when
    result = image_can_be_submitted_to_batch(
        batch_name="some",
        workspace_id="workspace",
        dataset_id="project",
        max_batch_images=10,
        api_key="api-key",
    )

    # then
    assert result is True


@mock.patch.object(accounting, "get_roboflow_labeling_jobs")
@mock.patch.object(accounting, "get_roboflow_labeling_batches")
def test_image_can_be_submitted_to_batch_when_found_matching_batches_with_labeling_jobs(
    get_roboflow_labeling_batches_mock: MagicMock,
    get_roboflow_labeling_jobs_mock: MagicMock,
) -> None:
    # given
    get_roboflow_labeling_batches_mock.return_value = {
        "batches": [
            {
                "name": "some",
                "numJobs": 1,
                "uploaded": {"_seconds": 1698060510, "_nanoseconds": 403000000},
                "images": 300,
                "id": "YYY",
            },
        ]
    }
    get_roboflow_labeling_jobs_mock.return_value = {
        "jobs": [
            {
                "sourceBatch": "XXX/YYY",
                "created": {"_seconds": 1697044995, "_nanoseconds": 375000000},
                "numImages": 161,
                "status": "complete",
                "unannotated": 0,
                "annotated": 161,
            },
            {
                "sourceBatch": "XXX/ZZZ",
                "created": {"_seconds": 1697044995, "_nanoseconds": 375000000},
                "numImages": 192,
                "status": "complete",
                "unannotated": 0,
                "annotated": 161,
            },
        ]
    }

    # when
    result = image_can_be_submitted_to_batch(
        batch_name="some",
        workspace_id="workspace",
        dataset_id="project",
        max_batch_images=461,
        api_key="api-key",
    )

    # then
    assert result is False
