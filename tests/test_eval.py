import json

from reap.eval import evalplus_attn_implementation, hf_evalplus_supported_for_eval


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
