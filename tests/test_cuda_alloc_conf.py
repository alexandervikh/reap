import os

from reap.model_util import _augment_cuda_alloc_conf


def test_adds_expandable_segments_when_empty():
    assert _augment_cuda_alloc_conf("") == "expandable_segments:True"


def test_appends_to_existing_conf():
    assert (
        _augment_cuda_alloc_conf("max_split_size_mb:128")
        == "max_split_size_mb:128,expandable_segments:True"
    )


def test_preserves_existing_expandable_segments_setting():
    assert (
        _augment_cuda_alloc_conf("expandable_segments:False")
        == "expandable_segments:False"
    )


def test_import_sets_env_var():
    # Importing reap.model_util must configure the allocator for the fused-MoE
    # load path that OOMs on transformers>=5.9 without expandable_segments.
    # Both the legacy and renamed env vars must be set so the setting is honored
    # regardless of the installed torch version.
    assert "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    assert "expandable_segments" in os.environ.get("PYTORCH_ALLOC_CONF", "")
