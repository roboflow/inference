import numpy as np

from inference_models.models.owlv2.cache import hash_reference_examples
from inference_models.models.owlv2.entities import (
    LazyReferenceExample,
    ReferenceBoundingBox,
)
from inference_models.models.owlv2.reference_dataset import (
    LazyImageWrapper,
    canonicalize_url_for_hashing,
)

OBJECT_URL = "https://storage.googleapis.com/bucket/workspace/image-1/original.jpg"
GCS_V4_SIGNED_URL_A = (
    f"{OBJECT_URL}?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=svc%40proj"
    "&X-Goog-Date=20260715T190000Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host"
    "&X-Goog-Signature=aaaa1111"
)
GCS_V4_SIGNED_URL_B = (
    f"{OBJECT_URL}?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=svc%40proj"
    "&X-Goog-Date=20260715T193000Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host"
    "&X-Goog-Signature=bbbb2222"
)


def wrap_reference(image) -> LazyImageWrapper:
    return LazyImageWrapper.init(
        image=image,
        allow_url_input=True,
        allow_non_https_url=False,
        allow_url_without_fqdn=False,
        whitelisted_domains=None,
        blacklisted_domains=None,
        allow_local_storage_access=False,
    )


def test_canonicalize_url_strips_query_when_gcs_v4_signature_present() -> None:
    # when
    result = canonicalize_url_for_hashing(reference=GCS_V4_SIGNED_URL_A)

    # then
    assert result == OBJECT_URL


def test_canonicalize_url_strips_query_for_other_known_signers() -> None:
    # given
    signed_variants = [
        f"{OBJECT_URL}?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc",
        f"{OBJECT_URL}?AWSAccessKeyId=AKIA123&Expires=1784150000&Signature=abc",
        f"{OBJECT_URL}?GoogleAccessId=svc%40proj&Expires=1784150000&Signature=abc",
        f"{OBJECT_URL}?sv=2024-05-04&se=2026-07-15T20%3A00%3A00Z&sig=abc",
    ]

    # when
    results = [canonicalize_url_for_hashing(reference=url) for url in signed_variants]

    # then
    assert all(result == OBJECT_URL for result in results)


def test_canonicalize_url_preserves_content_params_next_to_signature() -> None:
    # given - version/generation params select different bytes for the same
    # path and must survive canonicalization
    gcs_gen_1 = f"{OBJECT_URL}?generation=111&X-Goog-Signature=aaaa&X-Goog-Date=20260715T190000Z"
    gcs_gen_1_resigned = (
        f"{OBJECT_URL}?generation=111&X-Goog-Signature=bbbb&X-Goog-Date=20260715T193000Z"
    )
    gcs_gen_2 = f"{OBJECT_URL}?generation=222&X-Goog-Signature=aaaa&X-Goog-Date=20260715T190000Z"
    s3_version_a = f"{OBJECT_URL}?versionId=abc&X-Amz-Signature=aaaa"
    s3_version_b = f"{OBJECT_URL}?versionId=def&X-Amz-Signature=aaaa"

    # when
    results = [
        canonicalize_url_for_hashing(reference=url)
        for url in (gcs_gen_1, gcs_gen_1_resigned, gcs_gen_2, s3_version_a, s3_version_b)
    ]

    # then
    assert results[0] == f"{OBJECT_URL}?generation=111"
    assert results[0] == results[1]  # re-signing same generation -> same key
    assert results[2] == f"{OBJECT_URL}?generation=222"  # other generation distinct
    assert results[3] == f"{OBJECT_URL}?versionId=abc"
    assert results[3] != results[4]


def test_canonicalize_url_key_is_independent_of_param_order() -> None:
    # given
    url_a = f"{OBJECT_URL}?generation=111&versionId=abc&X-Goog-Signature=aaaa"
    url_b = f"{OBJECT_URL}?versionId=abc&generation=111&X-Goog-Signature=bbbb"

    # when
    result_a = canonicalize_url_for_hashing(reference=url_a)
    result_b = canonicalize_url_for_hashing(reference=url_b)

    # then
    assert result_a == result_b


def test_canonicalize_url_preserves_unsigned_query_parameters() -> None:
    # given
    versioned_url = f"{OBJECT_URL}?v=2"

    # when
    result = canonicalize_url_for_hashing(reference=versioned_url)

    # then - a bare query string may select different content, so it must
    # remain part of the cache identity
    assert result == versioned_url


def test_canonicalize_url_leaves_url_without_query_untouched() -> None:
    # when
    result = canonicalize_url_for_hashing(reference=OBJECT_URL)

    # then
    assert result == OBJECT_URL


def test_lazy_image_wrapper_hash_is_stable_across_re_signed_urls() -> None:
    # when
    hash_a = wrap_reference(GCS_V4_SIGNED_URL_A).get_hash()
    hash_b = wrap_reference(GCS_V4_SIGNED_URL_B).get_hash()
    hash_plain = wrap_reference(OBJECT_URL).get_hash()

    # then
    assert hash_a == hash_b == hash_plain


def test_lazy_image_wrapper_hash_differs_for_different_objects() -> None:
    # given
    other_object_signed = GCS_V4_SIGNED_URL_A.replace("image-1", "image-2")

    # when
    hash_a = wrap_reference(GCS_V4_SIGNED_URL_A).get_hash()
    hash_other = wrap_reference(other_object_signed).get_hash()

    # then
    assert hash_a != hash_other


def test_lazy_image_wrapper_hash_keeps_unsigned_query_variants_distinct() -> None:
    # when
    hash_v1 = wrap_reference(f"{OBJECT_URL}?v=1").get_hash()
    hash_v2 = wrap_reference(f"{OBJECT_URL}?v=2").get_hash()

    # then
    assert hash_v1 != hash_v2


def test_lazy_image_wrapper_hash_for_base64_reference_is_not_url_canonicalized() -> None:
    # given - base64 payloads are not URLs and must hash on their full content
    payload_a = "aGVsbG8/sig=looks-like-query"
    payload_b = "aGVsbG8/sig=other-value"

    # when
    hash_a = wrap_reference(payload_a).get_hash()
    hash_b = wrap_reference(payload_b).get_hash()

    # then
    assert hash_a != hash_b


def test_lazy_image_wrapper_hash_for_in_memory_image_unaffected() -> None:
    # given
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    # when
    hash_a = wrap_reference(image).get_hash()
    hash_b = wrap_reference(image.copy()).get_hash()

    # then
    assert hash_a == hash_b


def test_hash_reference_examples_stable_across_re_signed_urls() -> None:
    # given
    boxes = [ReferenceBoundingBox(x=10, y=10, w=5, h=5, cls="widget")]
    examples_a = [
        LazyReferenceExample(image=wrap_reference(GCS_V4_SIGNED_URL_A), boxes=boxes)
    ]
    examples_b = [
        LazyReferenceExample(image=wrap_reference(GCS_V4_SIGNED_URL_B), boxes=boxes)
    ]

    # when
    key_a = hash_reference_examples(reference_examples=examples_a)
    key_b = hash_reference_examples(reference_examples=examples_b)

    # then - the class-embeddings cache key must survive URL re-signing
    assert key_a == key_b
