"""Ordinary processing blocks; source clocks are preserved by the engine."""

from pydantic import Field
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
)


class CelsiusToFahrenheit(Block):
    """Convert each sample once and expose the persistent invocation count."""

    type = "source_demo/celsius_to_fahrenheit"
    outputs = {
        "fahrenheit": Output(FLOAT_KIND, source="celsius"),
        "count": Output(INTEGER_KIND, source="celsius"),
    }

    class Params(BlockParams):
        celsius: Ref(FLOAT_KIND) = Field(description="Temperature in degrees Celsius.")

    def __init__(self):
        self.count = 0

    def run(self, *, celsius: float) -> dict:
        """Convert a scalar without reading or writing its temporal metadata.

        Args:
            celsius: One acquired temperature.

        Returns:
            Fahrenheit value and number of calls on this block instance.
        """
        self.count += 1
        result = {"fahrenheit": celsius * 9 / 5 + 32, "count": self.count}

        return result


class IsBright(Block):
    """Derive a plain boolean condition from native tensor pixels."""

    type = "source_demo/is_bright"
    outputs = {"keep": Output(BOOLEAN_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Frame whose mean is inspected.")

    def run(self, *, image: ImageData) -> dict:
        """Decide whether the image should enter the nested resize branch.

        Args:
            image: Tensor-native RGB input frame.

        Returns:
            True for a mean channel value of at least 100.
        """
        result = {"keep": bool(image.tensor_image.float().mean() >= 100)}

        return result


class RecordTemperature(Block):
    """Record an authored output-free action in an injected local audit list."""

    type = "source_demo/record_temperature"

    class Params(BlockParams):
        fahrenheit: Ref(FLOAT_KIND) = Field(description="Converted value to record.")

    def __init__(self, *, audit):
        """Receive the host's audit resource without copying it.

        Args:
            audit: Mutable list used to observe exactly-once authored effects.
        """
        self.audit = audit

    def run(self, *, fahrenheit: float) -> dict:
        """Record the value even though no workflow output selects this action.

        Args:
            fahrenheit: Converted temperature from the shared processing step.

        Returns:
            No outputs; the audit entry is the action's observable result.
        """
        self.audit.append(fahrenheit)
        return {}
