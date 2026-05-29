import json

from reap.args import EvalArgs, ModelArgs
from reap.eval import evalplus_attn_implementation, hf_evalplus_supported_for_eval, run_evaluate


def test_evalplus_attn_implementation_avoids_glm_hub_kernel(monkeypatch):
    monkeypatch.delenv("REAP_EVALPLUS_ATTN_IMPLEMENTATION", raising=False)

    assert evalplus_attn_implementation("/tmp/GLM-5.1/pruned") == "eager"
    assert evalplus_attn_implementation("Qwen/Qwen3-30B-A3B") == "flash_attention_2"


def test_evalplus_attn_implementation_env_override(monkeypatch):
    monkeypatch.setenv("REAP_EVALPLUS_ATTN_IMPLEMENTATION", "sdpa")

    assert evalplus_attn_implementation("/tmp/GLM-5.1/pruned") == "sdpa"


def test_hf_evalplus_supported_for_eval_skips_glm(tmp_path):
    pruned_path = tmp_path / "reap-renorm_true-seed_42-0.25"
    pruned_path.mkdir()
    (pruned_path / "config.json").write_text(
        json.dumps({"architectures": ["GlmMoeDsaForCausalLM"]})
    )

    assert not hf_evalplus_supported_for_eval("/tmp/GLM-5.1/pruned")
    assert not hf_evalplus_supported_for_eval(str(pruned_path))
    assert hf_evalplus_supported_for_eval("Qwen/Qwen3-30B-A3B")


def test_run_math_skipped_without_server(tmp_path, monkeypatch):
    monkeypatch.setattr("reap.eval.torch.cuda.device_count", lambda: 0)
    monkeypatch.setattr("reap.eval.patched_model_map", lambda name: name)

    eval_args = EvalArgs(
        use_server=False,
        run_lm_eval=False,
        run_evalplus=False,
        run_livecodebench=False,
        run_wildbench=False,
        run_math=True,
        results_dir=str(tmp_path),
    )
    model_args = ModelArgs(model_name="/tmp/GLM-5.1/pruned")

    run_evaluate(model_args, tmp_path, eval_args, seed=42)
