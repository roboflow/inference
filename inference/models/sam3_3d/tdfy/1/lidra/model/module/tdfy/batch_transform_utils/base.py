from lidra.data.utils import build_batch_extractor, empty_mapping


class GenerativePreprocessing:
    """
    Base class for generative preprocessing.

    This class defines the interface for preprocessing batches in generative models.

    The workflow is as follows:
    1. `batch_preprocessing` method transforms the input batch, preparing it for the model
    2. `batch_encoder_input_mapping` defines which keys from the preprocessed batch should be
       extracted and passed to the encoder
    3. `batch_decoder_input_mapping` defines which keys from the preprocessed batch should be
       extracted and passed to the decoder
    4. `batch_conditions_mapping` defines which keys from the preprocessed batch should be
       extracted and used as conditions for the generative process

    Subclasses should implement the `batch_preprocessing` method and set appropriate
    values for the mapping properties based on their specific requirements.
    """

    @staticmethod
    def batch_preprocessing(batch):
        raise NotImplementedError

    batch_encoder_input_mapping = empty_mapping  # Keys to pluck from batch
    batch_decoder_input_mapping = empty_mapping  # Keys to pluck from batch
    batch_conditions_mapping = empty_mapping  # Keys to pluck from batch
