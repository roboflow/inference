"""Image-only embedding capability for classifiers with a recoverable head."""

from pathlib import Path
from threading import Lock

from inference_models.utils.onnx_embeddings import embedding_definition


class ImageEmbeddingModel:
    """Use the classifier's original preprocessing for features or raw logits."""

    def embed_images(self, images, output_type=None, **kwargs):
        return self.forward_embedding(
            self.pre_process(images, **kwargs), output_type=output_type
        )

    def _resolve_embedding_output_type(self, output_type):
        if output_type is None:
            output_type = getattr(self, "_embedding_output_type", "feature_vector")
        embedding_definition(output_type)
        return output_type


class TorchClassifierEmbeddings(ImageEmbeddingModel):
    def prepare_image_embeddings(self, output_type="feature_vector"):
        embedding_definition(output_type)
        self._embedding_output_type = output_type

    def forward_embedding(self, pre_processed_images, output_type=None):
        import torch

        output_type = self._resolve_embedding_output_type(output_type)
        with self._lock, torch.inference_mode():
            if output_type == "logits":
                return self._model.forward_logits(pre_processed_images)
            return self._model.forward_embedding(pre_processed_images)

    def get_embedding_info(self, output_type=None):
        output_type = self._resolve_embedding_output_type(output_type)
        return {
            "feature_definition": embedding_definition(output_type),
            "output_type": output_type,
            "normalization": "none",
        }

    @property
    def embedding_info(self):
        return self.get_embedding_info()


class OnnxClassifierEmbeddings(ImageEmbeddingModel):
    def prepare_image_embeddings(self, output_type=None):
        output_type = self._resolve_embedding_output_type(output_type)
        import onnxruntime

        from inference_models.utils.onnx_embeddings import prepare_classifier_embedding

        with self._session_thread_lock:
            if not hasattr(self, "_embedding_sessions"):
                self._embedding_sessions = {}
            if output_type in self._embedding_sessions:
                return self._embedding_sessions[output_type]
            path, info = prepare_classifier_embedding(
                self._embedding_source_path, output_type=output_type
            )
            session = onnxruntime.InferenceSession(
                path,
                sess_options=self._embedding_session_options,
                providers=embedding_providers(self._embedding_providers, path),
            )
            artifacts = [
                path,
                str(Path(path).with_name("embedding.json")),
            ]
            self.embedding_artifacts = list(
                dict.fromkeys(getattr(self, "embedding_artifacts", []) + artifacts)
            )
            self._embedding_sessions[output_type] = (session, info)
            return session, info

    def forward_embedding(self, pre_processed_images, output_type=None):
        from inference_models.models.common.onnx import (
            run_onnx_session_with_batch_size_limit,
        )

        session, _ = self.prepare_image_embeddings(output_type)
        with self._session_thread_lock:
            return run_onnx_session_with_batch_size_limit(
                session=session,
                inputs={self._input_name: pre_processed_images},
                min_batch_size=self._input_batch_size,
                max_batch_size=self._input_batch_size,
            )[0]

    def get_embedding_info(self, output_type=None):
        _, info = self.prepare_image_embeddings(output_type)
        return dict(info)

    @property
    def embedding_info(self):
        return self.get_embedding_info()


def embedding_providers(providers, path):
    result = []
    for provider in providers:
        if isinstance(provider, tuple):
            name, options = provider
            options = dict(options)
        else:
            name, options = provider, {}
        if name == "TensorrtExecutionProvider":
            options["trt_engine_cache_path"] = str(Path(path).parent)
        result.append((name, options))
    return result


def load_classifier_session(
    source_path, providers, required_capabilities=None, output_type="feature_vector"
):
    import onnxruntime

    from inference_models.utils.onnx_embeddings import prepare_classifier_embedding

    info = None
    path = source_path
    if "image_embeddings" in (required_capabilities or ()):
        path, info = prepare_classifier_embedding(source_path, output_type=output_type)
        providers = embedding_providers(providers, path)
    return onnxruntime.InferenceSession(path, providers=providers), info, path


class LazyClassifierSession:
    """Embedding-only loads allocate the classifier session only if later requested."""

    def __init__(self, source_path, providers, session_options):
        self._source_path = source_path
        self._providers = providers
        self._session_options = session_options
        self._session = None
        self._lock = Lock()

    def __getattr__(self, name):
        import onnxruntime

        with self._lock:
            if self._session is None:
                self._session = onnxruntime.InferenceSession(
                    self._source_path,
                    providers=self._providers,
                    sess_options=self._session_options,
                )
        return getattr(self._session, name)


def configure_onnx_embeddings(
    model,
    source_path,
    providers,
    embedding_info=None,
    embedding_path=None,
    output_type="feature_vector",
):
    model._embedding_output_type = output_type
    model._embedding_source_path = source_path
    model._embedding_providers = providers
    model._embedding_session_options = model._session.get_session_options()
    if embedding_info is not None:
        model._embedding_sessions = {output_type: (model._session, embedding_info)}
        path = embedding_path
        model.embedding_artifacts = [path, str(Path(path).with_name("embedding.json"))]
        model._session = LazyClassifierSession(
            source_path, providers, model._embedding_session_options
        )
    return model
