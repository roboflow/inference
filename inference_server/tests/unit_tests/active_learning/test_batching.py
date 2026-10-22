from datetime import date, datetime
from unittest.mock import MagicMock

from inference_server.active_learning.batching import generate_batch_name
from inference_server.active_learning.entities import BatchReCreationInterval


def test_generate_batch_name_when_single_batch_should_be_used_for_all_images() -> None:
    configuration = MagicMock()
    configuration.batches_name_prefix = "active_learning_batch"
    configuration.batch_recreation_interval = BatchReCreationInterval.NEVER

    result = generate_batch_name(configuration=configuration)

    assert result == "active_learning_batch"


def test_generate_batch_name_when_time_based_created_batches_to_be_used() -> None:
    configuration = MagicMock()
    configuration.batches_name_prefix = "active_learning_batch"
    configuration.batch_recreation_interval = BatchReCreationInterval.DAILY
    today_timestamp = date.today()

    result = generate_batch_name(configuration=configuration)
    result_time_stamp = datetime.strptime(
        "_".join(result.split("_")[-3:]), "%Y_%m_%d"
    ).date()

    assert result.startswith("active_learning_batch")
    assert today_timestamp == result_time_stamp
