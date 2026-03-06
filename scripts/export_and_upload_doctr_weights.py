#!/usr/bin/env python3
"""
Script to export doctr 0.11+ weights and upload them to GCS.

This script:
1. Loads pretrained doctr models (detection and recognition)
2. Exports their state_dict to local .pt files
3. Uploads them to GCS buckets

Requirements:
- python-doctr[torch]>=0.11.0
- torch
- google-cloud-storage

Usage:
    python export_and_upload_doctr_weights.py --env staging
    python export_and_upload_doctr_weights.py --env production
    python export_and_upload_doctr_weights.py --export-only  # Just export locally, no upload
"""

import argparse
import os
import tempfile

import torch
from doctr.models import detection_predictor, recognition_predictor

# Detection models supported by inference_models
DETECTION_MODELS = [
    "fast_base",
    "fast_small",
    "fast_tiny",
    "db_resnet50",
    "db_resnet34",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
]

# Recognition models supported by inference_models
RECOGNITION_MODELS = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "master",
    "sar_resnet31",
    "vitstr_small",
    "vitstr_base",
    "parseq",
]

GCS_BUCKETS = {
    "staging": "roboflow-staging-core-models",
    "production": "roboflow-platform-core-models",
}


def export_weights(output_dir: str) -> dict:
    """Export all doctr model weights to local directory."""
    os.makedirs(output_dir, exist_ok=True)

    exported_files = {
        "detection": {},
        "recognition": {},
    }

    print("Exporting detection models...")
    for model_name in DETECTION_MODELS:
        print(f"  Loading {model_name}...")
        try:
            model = detection_predictor(
                arch=model_name,
                pretrained=True,
                pretrained_backbone=True,
            )
            out_path = os.path.join(output_dir, f"det_{model_name}.pt")
            torch.save(model.model.state_dict(), out_path)
            exported_files["detection"][model_name] = out_path
            print(f"    ✓ Saved to {out_path}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    print("\nExporting recognition models...")
    for model_name in RECOGNITION_MODELS:
        print(f"  Loading {model_name}...")
        try:
            model = recognition_predictor(
                arch=model_name,
                pretrained=True,
                pretrained_backbone=True,
            )
            out_path = os.path.join(output_dir, f"rec_{model_name}.pt")
            torch.save(model.model.state_dict(), out_path)
            exported_files["recognition"][model_name] = out_path
            print(f"    ✓ Saved to {out_path}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    return exported_files


def upload_to_gcs(exported_files: dict, environment: str) -> None:
    """Upload exported weights to GCS."""
    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
        print("\nAlternatively, use gsutil commands below:\n")
        print_gsutil_commands(exported_files, environment)
        return

    bucket_name = GCS_BUCKETS[environment]
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    print(f"\nUploading to gs://{bucket_name}/doctr-new-inference-v2/...")

    # Upload detection models
    for model_name, local_path in exported_files["detection"].items():
        blob_path = f"doctr-new-inference-v2/det-models/{model_name}.pt"
        blob = bucket.blob(blob_path)

        if blob.exists():
            print(f"  Already exists: {blob_path}")
        else:
            print(f"  Uploading {model_name} -> {blob_path}")
            blob.upload_from_filename(local_path)
            print(f"    ✓ Done")

    # Upload recognition models
    for model_name, local_path in exported_files["recognition"].items():
        blob_path = f"doctr-new-inference-v2/rec-models/{model_name}.pt"
        blob = bucket.blob(blob_path)

        if blob.exists():
            print(f"  Already exists: {blob_path}")
        else:
            print(f"  Uploading {model_name} -> {blob_path}")
            blob.upload_from_filename(local_path)
            print(f"    ✓ Done")

    print(f"\n✓ All weights uploaded to gs://{bucket_name}/doctr-new-inference-v2/")


def print_gsutil_commands(exported_files: dict, environment: str) -> None:
    """Print gsutil commands for manual upload."""
    bucket_name = GCS_BUCKETS[environment]

    print("# Detection models")
    for model_name, local_path in exported_files["detection"].items():
        gcs_path = f"gs://{bucket_name}/doctr-new-inference-v2/det-models/{model_name}.pt"
        print(f"gsutil cp {local_path} {gcs_path}")

    print("\n# Recognition models")
    for model_name, local_path in exported_files["recognition"].items():
        gcs_path = f"gs://{bucket_name}/doctr-new-inference-v2/rec-models/{model_name}.pt"
        print(f"gsutil cp {local_path} {gcs_path}")


def main():
    parser = argparse.ArgumentParser(description="Export and upload doctr weights to GCS")
    parser.add_argument(
        "--env",
        type=str,
        choices=["staging", "production"],
        default="staging",
        help="Target environment (staging or production)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export weights locally, don't upload to GCS",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to export weights to (default: temp directory)",
    )
    parser.add_argument(
        "--print-gsutil",
        action="store_true",
        help="Print gsutil commands instead of using Python GCS client",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        cleanup = False
    else:
        output_dir = "/tmp/doctr_weights_v2"
        os.makedirs(output_dir, exist_ok=True)
        cleanup = False  # Keep temp files for manual upload if needed

    print(f"Output directory: {output_dir}")
    print(f"Target environment: {args.env}")
    print()

    # Export weights
    exported_files = export_weights(output_dir)

    print(f"\n✓ Exported {len(exported_files['detection'])} detection models")
    print(f"✓ Exported {len(exported_files['recognition'])} recognition models")

    if args.export_only:
        print(f"\nWeights exported to: {output_dir}")
        print("\nTo upload manually, run:")
        print_gsutil_commands(exported_files, args.env)
        return

    if args.print_gsutil:
        print("\ngsutil commands for upload:")
        print_gsutil_commands(exported_files, args.env)
        return

    # Upload to GCS
    upload_to_gcs(exported_files, args.env)


if __name__ == "__main__":
    main()
