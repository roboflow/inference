"""Project-local immutable publication storage with transactional lifecycle control."""

from typing import Any

from inference_model_manager.memory_profiles.contracts import Catalog, Policy, Template
from inference_model_manager.memory_profiles.matching import (
    MemoryProfileError,
    Snapshot,
    content_digest,
    validate_snapshot,
)


def _transaction(client, operation, *, read_only=False):
    from google.api_core.exceptions import (
        DeadlineExceeded,
        Forbidden,
        PermissionDenied,
        ServiceUnavailable,
        Unauthenticated,
    )
    from google.cloud import firestore

    transaction = client.transaction(read_only=read_only)
    try:
        result = firestore.transactional(operation)(transaction)
    except (Forbidden, PermissionDenied, Unauthenticated) as error:
        raise MemoryProfileError(
            "memory_access_denied", "Publication metadata access denied"
        ) from error
    except (DeadlineExceeded, ServiceUnavailable) as error:
        raise MemoryProfileError(
            "memory_lookup_unavailable",
            "Publication metadata service unavailable",
            retryable=True,
        ) from error

    return result


class FirestoreCatalog:
    """Read and publish memory metadata in one explicitly selected project.

    Credentials must already authorize this project. No default-project fallback,
    document cache, offline snapshot or cross-project activation is implemented.

    Args:
        client: Authenticated google.cloud.firestore.Client for the selected project.
        project: Required project identity, checked against the client.
    """

    def __init__(self, client: Any, *, project: str):
        if not project or client.project != project:
            raise ValueError("Firestore client project differs from configured project")
        self.client = client
        self.project = project

    def _ref(self, collection, identity):
        if not identity or "/" in identity:
            raise ValueError("Invalid document ID")
        return self.client.collection(collection).document(identity)

    def _template_ref(self, identity, version):
        if not version or "/" in version:
            raise ValueError("Invalid template version")
        return (
            self._ref("memory_profile_templates", identity)
            .collection("versions")
            .document(version)
        )

    def _read(self, ref, transaction):
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            raise MemoryProfileError(
                "memory_publication_invalid", "Required publication document is absent"
            )
        return snapshot.to_dict()

    def _snapshot(self, transaction, *, catalog_id, policy_id, revision):
        catalog = Catalog.model_validate(
            self._read(self._ref("memory_profile_catalogs", catalog_id), transaction)
        )
        policy = Policy.model_validate(
            self._read(self._ref("memory_profile_policies", policy_id), transaction)
        )
        if catalog.id != catalog_id or policy.id != policy_id:
            raise MemoryProfileError(
                "memory_publication_invalid", "Document ID differs from content"
            )
        templates = tuple(
            Template.model_validate(
                self._read(self._template_ref(r.id, r.version), transaction)
            )
            for r in catalog.templates
        )
        result = Snapshot(self.project, revision, catalog, policy, templates)
        validate_snapshot(result)
        return result

    def read_active(self, deployment_group: str) -> Snapshot:
        """Read controls, readiness, policy and immutable contents at one snapshot.

        Args:
            deployment_group: Trusted server configuration, never request input.

        Returns:
            A verified publication snapshot, with no dedicated caching.

        Raises:
            MemoryProfileError: If active publication is absent, withdrawn or corrupt.
            Exception: Firestore authorization/transient errors propagate to the caller.
        """

        def read(transaction):
            control = self._read(
                self._ref("memory_profile_controls", deployment_group), transaction
            )
            publication = self._read(
                self._ref("memory_profile_publications", control["catalog_id"]),
                transaction,
            )
            if (
                publication["state"] != "ready"
                or publication["policy_id"] != control["policy_id"]
            ):
                raise MemoryProfileError(
                    "memory_publication_invalid",
                    "Active publication is not ready under the selected policy",
                )
            snapshot = self._snapshot(
                transaction,
                catalog_id=control["catalog_id"],
                policy_id=control["policy_id"],
                revision=control["revision"],
            )
            if publication["catalog_sha256"] != content_digest(
                snapshot.catalog
            ) or publication["policy_sha256"] != content_digest(snapshot.policy):
                raise MemoryProfileError(
                    "memory_publication_invalid", "Publication content hash mismatch"
                )
            return snapshot

        try:
            result = _transaction(self.client, read, read_only=True)
        except MemoryProfileError:
            raise
        except (ValueError, KeyError, TypeError) as error:
            raise MemoryProfileError(
                "memory_publication_invalid", "Malformed publication metadata"
            ) from error
        return result

    def stage(self, snapshot: Snapshot) -> None:
        """Create immutable objects and an inactive draft; identical repeats are safe.

        Args:
            snapshot: Evidence-validated contents produced by publication tooling.

        Raises:
            MemoryProfileError: If references are invalid or an ID already has different contents.
        """
        if snapshot.project != self.project:
            raise ValueError("Cross-project publication is forbidden")
        validate_snapshot(snapshot)
        documents = [
            (self._template_ref(t.id, t.version), t.model_dump(mode="json"))
            for t in snapshot.templates
        ]
        documents += [
            (
                self._ref("memory_profile_catalogs", snapshot.catalog.id),
                snapshot.catalog.model_dump(mode="json"),
            ),
            (
                self._ref("memory_profile_policies", snapshot.policy.id),
                snapshot.policy.model_dump(mode="json"),
            ),
        ]
        # Bound both transaction size and document size before any mutation.
        import json

        if len(documents) > 450 or any(
            len(json.dumps(value).encode()) > 800_000 for _, value in documents
        ):
            raise ValueError(
                "Catalog exceeds M1 publication size limit; split the catalog"
            )

        def write(transaction):
            existing = [
                (ref, value, ref.get(transaction=transaction))
                for ref, value in documents
            ]
            pub_ref = self._ref("memory_profile_publications", snapshot.catalog.id)
            publication = pub_ref.get(transaction=transaction)
            expected = {
                "state": "draft",
                "policy_id": snapshot.policy.id,
                "catalog_sha256": content_digest(snapshot.catalog),
                "policy_sha256": content_digest(snapshot.policy),
            }
            for _, value, old in existing:
                if old.exists and old.to_dict() != value:
                    raise MemoryProfileError(
                        "memory_publication_conflict",
                        "Immutable object ID already has different contents",
                    )
            if publication.exists and any(
                publication.to_dict().get(k) != v
                for k, v in expected.items()
                if k != "state"
            ):
                raise MemoryProfileError(
                    "memory_publication_conflict",
                    "Publication already references different content",
                )
            for ref, value, old in existing:
                if not old.exists:
                    transaction.create(ref, value)
            if not publication.exists:
                transaction.create(pub_ref, expected)

        _transaction(self.client, write)

    def mark_ready(self, catalog_id: str, *, approval: str) -> None:
        """Record explicit approval after validating draft contents again.

        Args:
            catalog_id: Project-local draft publication.
            approval: Nonempty review/approval reference supplied by the publisher.

        Raises:
            MemoryProfileError: If publication is withdrawn or contents are invalid.
            ValueError: If no approval reference is supplied.
        """
        if not approval.strip():
            raise ValueError("Approval reference required")

        def ready(transaction):
            ref = self._ref("memory_profile_publications", catalog_id)
            publication = self._read(ref, transaction)
            if publication["state"] not in {"draft", "ready"}:
                raise MemoryProfileError(
                    "memory_publication_conflict",
                    "Withdrawn publication cannot be reactivated",
                )
            snapshot = self._snapshot(
                transaction,
                catalog_id=catalog_id,
                policy_id=publication["policy_id"],
                revision=0,
            )
            if publication["catalog_sha256"] != content_digest(
                snapshot.catalog
            ) or publication["policy_sha256"] != content_digest(snapshot.policy):
                raise MemoryProfileError(
                    "memory_publication_invalid", "Content changed after staging"
                )
            transaction.update(ref, {"state": "ready", "approval": approval})

        _transaction(self.client, ready)

    def activate(
        self, catalog_id: str, *, deployment_group: str, expected_revision: int
    ) -> None:
        """Atomically activate an approved catalog/policy pair using compare-and-swap.

        Args:
            catalog_id: Ready publication to activate or roll back to.
            deployment_group: Server deployment control document.
            expected_revision: Last observed control revision; zero for initial activation.

        Raises:
            MemoryProfileError: If readiness or revision changed concurrently.
        """

        def activate(transaction):
            publication = self._read(
                self._ref("memory_profile_publications", catalog_id), transaction
            )
            control_ref = self._ref("memory_profile_controls", deployment_group)
            control = control_ref.get(transaction=transaction)
            revision = control.to_dict()["revision"] if control.exists else 0
            if revision != expected_revision or publication["state"] != "ready":
                raise MemoryProfileError(
                    "memory_publication_conflict",
                    "Control revision changed or publication is not ready",
                )
            transaction.set(
                control_ref,
                {
                    "catalog_id": catalog_id,
                    "policy_id": publication["policy_id"],
                    "revision": revision + 1,
                },
            )

        _transaction(self.client, activate)

    def withdraw(self, catalog_id: str) -> None:
        """Prevent future lookups from using a project-local publication.

        Args:
            catalog_id: Publication to withdraw; immutable evidence is retained.
        """

        def withdraw(transaction):
            ref = self._ref("memory_profile_publications", catalog_id)
            self._read(ref, transaction)
            transaction.update(ref, {"state": "withdrawn"})

        _transaction(self.client, withdraw)
