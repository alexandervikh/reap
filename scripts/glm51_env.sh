#!/usr/bin/env bash
# Shared GLM-5.1 paths and load-mode defaults (source from other scripts).
# GLM51_VARIANT: bf16 (default) | fp8

: "${REPO_ROOT:?REPO_ROOT must be set before sourcing glm51_env.sh}"

export GLM51_VARIANT="${GLM51_VARIANT:-bf16}"
export GLM51_LOAD_MODE="${GLM51_LOAD_MODE:-fast_ram}"
export HF_HOME="${HF_HOME:-/tmp/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "$GLM51_VARIANT" in
  bf16)
    GLM51_HF_ID="zai-org/GLM-5.1"
    GLM51_LOCAL_DIR="/tmp/GLM-5.1"
    if [[ -n "${GLM51_DOWNLOAD_DIR:-}" ]]; then
      GLM51_ARTIFACTS_DIR="$GLM51_DOWNLOAD_DIR"
    elif [[ -d "$GLM51_LOCAL_DIR" ]]; then
      GLM51_ARTIFACTS_DIR="$GLM51_LOCAL_DIR"
    else
      GLM51_ARTIFACTS_DIR="${REPO_ROOT}/artifacts/models/GLM-5.1"
    fi
    ;;
  fp8)
    echo "glm51_env: GLM51_VARIANT=fp8 is deprecated for this rollout; use bf16." >&2
    GLM51_HF_ID="zai-org/GLM-5.1-FP8"
    GLM51_LOCAL_DIR="/tmp/GLM-5.1-FP8"
    if [[ -L "${REPO_ROOT}/artifacts/models/GLM-5.1-FP8" ]]; then
      GLM51_ARTIFACTS_DIR="$(readlink -f "${REPO_ROOT}/artifacts/models/GLM-5.1-FP8")"
    elif [[ -d "$GLM51_LOCAL_DIR" ]]; then
      GLM51_ARTIFACTS_DIR="$GLM51_LOCAL_DIR"
    else
      GLM51_ARTIFACTS_DIR="${REPO_ROOT}/artifacts/models/GLM-5.1-FP8"
    fi
    ;;
  *)
    echo "glm51_env: unknown GLM51_VARIANT=$GLM51_VARIANT (use bf16 or fp8)" >&2
    exit 1
    ;;
esac

export GLM51_HF_ID GLM51_ARTIFACTS_DIR GLM51_LOCAL_DIR

_resolve_glm51_model_path() {
  if [[ -n "${MODEL:-}" ]]; then
    echo "$MODEL"
    return
  fi
  if [[ "$GLM51_LOAD_MODE" == "local" || "$GLM51_LOAD_MODE" == "fast_ram" || "$GLM51_LOAD_MODE" == "cpu_dispatch" ]]; then
    if [[ -d "$GLM51_LOCAL_DIR" ]]; then
      echo "$GLM51_LOCAL_DIR"
      return
    fi
  fi
  if [[ -d "$GLM51_ARTIFACTS_DIR" ]]; then
    echo "$GLM51_ARTIFACTS_DIR"
    return
  fi
  echo "$GLM51_HF_ID"
}

export GLM51_MODEL_PATH
GLM51_MODEL_PATH="$(_resolve_glm51_model_path)"
