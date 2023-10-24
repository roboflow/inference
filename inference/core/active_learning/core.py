from typing import Any

import numpy as np


ImageIdentifier = str


class ActiveLearningMiddleware:

    def register_image(self, image_id: ImageIdentifier, image: np.ndarray) -> None:
        pass

    def register_prediction(
        self,
        image_id: ImageIdentifier,
        prediction_type: str,
        prediction: Any,
    ) -> None:
        pass
