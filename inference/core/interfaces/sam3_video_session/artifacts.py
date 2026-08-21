import json
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

from inference.core.interfaces.sam3_video_session.entities import (
    ARTIFACT_CHUNK_CONTENT_TYPE,
    Sam3VideoTimeBase,
    validated_app_base_url,
)
from inference.core.utils.requests import api_key_safe_raise_for_status


class ArtifactWriter:
    def __init__(
        self,
        *,
        app_base_url: str,
        video_id: str,
        workspace_id: str,
        dataset_id: str,
        revision_id: str,
        api_key: str,
        timeout_seconds: float = 60.0,
    ):
        if not api_key:
            raise ValueError("api_key is required to write SAM3 video artifacts")
        self._app_base_url = validated_app_base_url(app_base_url)
        self._video_id = video_id
        self._workspace_id = workspace_id
        self._dataset_id = dataset_id
        self._revision_id = revision_id
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def checkpoint_chunk(
        self,
        *,
        track_id: str,
        chunk_index: int,
        total_chunks: int,
        samples: List[Dict[str, Any]],
    ) -> None:
        mint_url = urljoin(
            self._app_base_url,
            f"query/video/{self._video_id}/tracks/{track_id}/artifact-chunks/upload-url",
        )
        mint_response = requests.post(
            mint_url,
            json={
                "workspaceId": self._workspace_id,
                "datasetId": self._dataset_id,
                "revisionId": self._revision_id,
                "chunkIndex": chunk_index,
                "totalChunks": total_chunks,
            },
            headers=self._auth_headers(),
            timeout=self._timeout_seconds,
        )
        api_key_safe_raise_for_status(response=mint_response)
        upload_url = mint_response.json().get("uploadUrl")
        if not upload_url:
            raise RuntimeError("Roboflow did not return an artifact upload URL.")
        put_response = requests.put(
            upload_url,
            data=json.dumps({"samples": samples}).encode("utf-8"),
            headers={"Content-Type": ARTIFACT_CHUNK_CONTENT_TYPE},
            timeout=self._timeout_seconds,
        )
        api_key_safe_raise_for_status(response=put_response)

    def commit_revision(
        self,
        *,
        track_id: str,
        start_frame_index: int,
        end_frame_index: int,
        start_pts: int,
        end_pts: int,
        video_time_base: Sam3VideoTimeBase,
        class_name: str,
        tracker_id: int,
        sample_count: int,
        chunk_count: int,
    ) -> None:
        url = urljoin(
            self._app_base_url,
            f"query/video/{self._video_id}/tracks/{track_id}/artifact-revisions/{self._revision_id}/commit",
        )
        response = requests.post(
            url,
            json={
                "workspaceId": self._workspace_id,
                "datasetId": self._dataset_id,
                "startFrameIndex": start_frame_index,
                "endFrameIndex": end_frame_index,
                "startPts": start_pts,
                "endPts": end_pts,
                "videoTimeBase": {
                    "numerator": video_time_base.numerator,
                    "denominator": video_time_base.denominator,
                },
                "className": class_name,
                "trackerId": tracker_id,
                "sampleCount": sample_count,
                "chunkCount": chunk_count,
            },
            headers=self._auth_headers(),
            timeout=self._timeout_seconds,
        )
        api_key_safe_raise_for_status(response=response)


class TrackAccumulator:
    def __init__(self, track_id: str, class_name: str, tracker_id: int):
        self.track_id = track_id
        self.class_name = class_name
        self.tracker_id = tracker_id
        self.start_frame_index: Optional[int] = None
        self.end_frame_index: Optional[int] = None
        self.start_pts: Optional[int] = None
        self.end_pts: Optional[int] = None
        self.pending_samples: List[Dict[str, Any]] = []
        self.flushed_chunks = 0
        self.sample_count = 0

    def add_sample(self, sample: Dict[str, Any], frame_index: int, pts: int) -> None:
        if self.start_frame_index is None:
            self.start_frame_index = frame_index
            self.start_pts = pts
        self.end_frame_index = (
            frame_index
            if self.end_frame_index is None
            else max(self.end_frame_index, frame_index)
        )
        self.end_pts = pts if self.end_pts is None else max(self.end_pts, pts)
        self.class_name = str(sample.get("className") or self.class_name)
        self.pending_samples.append(sample)
        self.sample_count += 1

    def flush_ready(
        self,
        writer: ArtifactWriter,
        *,
        chunk_sample_size: int,
        is_final: bool,
    ) -> int:
        flushed = 0
        while self.pending_samples and (
            is_final or len(self.pending_samples) >= chunk_sample_size
        ):
            if is_final:
                samples = self.pending_samples
                self.pending_samples = []
            else:
                samples = self.pending_samples[:chunk_sample_size]
                self.pending_samples = self.pending_samples[chunk_sample_size:]
            if not samples:
                break
            writer.checkpoint_chunk(
                track_id=self.track_id,
                chunk_index=self.flushed_chunks,
                total_chunks=self.flushed_chunks + 1,
                samples=samples,
            )
            self.flushed_chunks += 1
            flushed += 1
        return flushed

    def commit(
        self, writer: ArtifactWriter, video_time_base: Sam3VideoTimeBase
    ) -> None:
        if self.sample_count == 0 or self.start_frame_index is None:
            return
        writer.commit_revision(
            track_id=self.track_id,
            start_frame_index=self.start_frame_index,
            end_frame_index=self.end_frame_index or self.start_frame_index,
            start_pts=self.start_pts or 0,
            end_pts=self.end_pts or 0,
            video_time_base=video_time_base,
            class_name=self.class_name,
            tracker_id=self.tracker_id,
            sample_count=self.sample_count,
            chunk_count=max(1, self.flushed_chunks),
        )
