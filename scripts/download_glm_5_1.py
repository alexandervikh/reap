"""Download GLM-5.1 (BF16) or GLM-5.1-FP8 for local REAP experiments."""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

VARIANTS = {
    "bf16": ("zai-org/GLM-5.1", "GLM-5.1"),
    "fp8": ("zai-org/GLM-5.1-FP8", "GLM-5.1-FP8"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="bf16",
        help="bf16 = zai-org/GLM-5.1; fp8 = zai-org/GLM-5.1-FP8",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Override output directory (default: artifacts/models/<name> or GLM51_DOWNLOAD_DIR)",
    )
    args = parser.parse_args()
    repo_id, dir_name = VARIANTS[args.variant]
    if args.dest:
        model_dir = os.path.abspath(args.dest)
    elif os.environ.get("GLM51_DOWNLOAD_DIR"):
        model_dir = os.path.abspath(os.environ["GLM51_DOWNLOAD_DIR"])
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.normpath(
            os.path.join(script_dir, os.pardir, "artifacts", "models")
        )
        model_dir = os.path.join(artifacts_dir, dir_name)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=model_dir,
    )
    print(f"Downloaded {repo_id} to {model_dir}")


if __name__ == "__main__":
    main()
