from inference.core.active_learning.entities import (
    ActiveLearningConfiguration,
    BatchReCreationInterval,
)
from inference.core.active_learning.utils import (
    generate_start_timestamp_for_this_month,
    generate_start_timestamp_for_this_week,
    generate_today_timestamp,
)

RECREATION_INTERVAL2TIMESTAMP_GENERATOR = {
    BatchReCreationInterval.DAILY: generate_today_timestamp,
    BatchReCreationInterval.WEEKLY: generate_start_timestamp_for_this_week,
    BatchReCreationInterval.MONTHLY: generate_start_timestamp_for_this_month,
}


def generate_batch_name(configuration: ActiveLearningConfiguration) -> str:
    batch_name = configuration.batches_name_prefix
    if configuration.batch_recreation_interval is BatchReCreationInterval.NEVER:
        return batch_name
    timestamp_generator = RECREATION_INTERVAL2TIMESTAMP_GENERATOR[
        configuration.batch_recreation_interval
    ]
    timestamp = timestamp_generator()
    return f"{batch_name}_{timestamp}"
