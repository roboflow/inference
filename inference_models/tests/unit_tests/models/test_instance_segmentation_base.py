from inference_models.models.base.instance_segmentation import (
    InstanceSegmentationModel,
)


class _StubSegModel(InstanceSegmentationModel):
    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls()

    @property
    def class_names(self):
        return ["a"]

    @property
    def supported_mask_formats(self):
        return ()

    def pre_process(self, images, **kwargs):
        return images, None

    def forward(self, pre_processed_images, **kwargs):
        return pre_processed_images

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        return model_results


def test_max_batch_size_is_a_property():
    assert isinstance(InstanceSegmentationModel.max_batch_size, property)


def test_max_batch_size_reads_underlying_attribute():
    model = _StubSegModel()
    model._max_batch_size = 8
    assert model.max_batch_size == 8


def test_max_batch_size_defaults_to_none():
    assert _StubSegModel().max_batch_size is None
