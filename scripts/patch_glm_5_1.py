"""Download GLM-5.1-FP8 for local REAP experiments (in-tree transformers, no patching)."""

import os

from huggingface_hub import snapshot_download


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.normpath(
        os.path.join(script_dir, os.pardir, "artifacts", "models")
    )
    model_dir = os.path.join(artifacts_dir, "GLM-5.1-FP8")

    snapshot_download(
        repo_id="zai-org/GLM-5.1-FP8",
        repo_type="model",
        local_dir=model_dir,
    )
    print(f"Downloaded zai-org/GLM-5.1-FP8 to {model_dir}")


if __name__ == "__main__":
    main()
