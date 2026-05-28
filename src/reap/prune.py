from __future__ import annotations
import time
import logging
import dataclasses
import pathlib
import time
from typing import Any
import gc
import yaml

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from accelerate.utils import set_seed
from accelerate.hooks import remove_hook_from_module


from reap.main import record_activations, smoke_test, create_results_directory, dump_args_to_yaml
from reap.args import (
    ReapArgs,
    ModelArgs,
    EvalArgs,
    PruneArgs,
    ObserverArgs,
    DatasetArgs,
    ClusterArgs,
)
from reap.data import DATASET_REGISTRY
from reap.cluster import (
    get_penalty_vector,
    hierarchical_clustering,
    dynamic_frequency_penalized_clustering,
)
from reap.model_util import (
    get_moe,
    assert_merge,
    MODEL_ATTRS,
    patched_model_map,
    get_super_expert_indices,
    load_causal_lm_for_prune,
    get_model_device,
)
from reap.eval import run_evaluate
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def _expert_count_attr_names(model_attrs: dict[str, Any]) -> list[str]:
    """Ordered unique MODEL_ATTRS names for runtime expert-count fields."""
    names: list[str] = []
    for key in ("moe_num_experts", "num_experts", "fused_experts_count"):
        attr = model_attrs.get(key)
        if attr and attr not in names:
            names.append(attr)
    return names


def _set_module_expert_counts(
    module: Any,
    model_attrs: dict[str, Any],
    retained_count: int,
    *,
    required_attrs: tuple[str, ...] = (),
) -> None:
    """Set every mapped expert-count attribute present on ``module``."""
    for attr in _expert_count_attr_names(model_attrs):
        if hasattr(module, attr):
            setattr(module, attr, retained_count)
    for attr in required_attrs:
        if not hasattr(module, attr):
            raise AttributeError(
                f"{module.__class__.__name__} missing required expert-count "
                f"attribute {attr!r} (MODEL_ATTRS)"
            )
        setattr(module, attr, retained_count)


def _set_fused_expert_counts(moe, model_attrs: dict[str, Any], retained_count: int) -> None:
    """Update fused MoE expert counts using MODEL_ATTRS only.

    ``model_attrs["num_experts"]`` is the HuggingFace **config** key (also applied to
    ``model.config`` after pruning). The MoE block and experts submodule may use
    different runtime names (e.g. Llama-4 config ``num_local_experts`` vs
    ``moe.num_experts``; GLM config ``n_routed_experts`` vs ``experts.num_experts``).
    """
    if not model_attrs.get("moe_num_experts"):
        raise KeyError(
            "Fused MoE models require MODEL_ATTRS['moe_num_experts'] "
            f"(missing for {moe.__class__.__name__})"
        )

    experts = moe.experts
    experts_attrs: list[str] = []
    if fused_attr := model_attrs.get("fused_experts_count"):
        experts_attrs.append(fused_attr)
    for attr in _expert_count_attr_names(model_attrs):
        if attr not in experts_attrs:
            experts_attrs.append(attr)
    for attr in experts_attrs:
        if hasattr(experts, attr):
            setattr(experts, attr, retained_count)

    _set_module_expert_counts(
        moe,
        model_attrs,
        retained_count,
        required_attrs=(model_attrs["moe_num_experts"],),
    )


def _set_fused_router_counts(router: Any, model_attrs: dict[str, Any], retained_count: int) -> None:
    """Update router/gate expert-count metadata after fused weight pruning."""
    _set_module_expert_counts(router, model_attrs, retained_count)
    if hasattr(router, "out_features"):
        router.out_features = retained_count


def _module_name_for_prune(model: torch.nn.Module, module: torch.nn.Module) -> str:
    for name, candidate in model.named_modules():
        if candidate is module:
            return name
    raise ValueError(f"Could not locate {module.__class__.__name__} in model modules")


def _materialized_tensor_for_prune(
    model: torch.nn.Module,
    module_name: str,
    attr_name: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if tensor.device.type != "meta":
        return tensor.detach()

    from transformers.integrations.accelerate import load_offloaded_parameter

    full_name = f"{module_name}.{attr_name}" if module_name else attr_name
    return load_offloaded_parameter(model, full_name).detach()


def _index_select_tensor_attr_for_prune(
    model: torch.nn.Module,
    module: torch.nn.Module,
    module_name: str,
    attr_name: str,
    retained_indices: list[int],
    *,
    dim: int = 0,
) -> None:
    tensor = getattr(module, attr_name)
    source = _materialized_tensor_for_prune(model, module_name, attr_name, tensor)
    index = torch.as_tensor(retained_indices, device=source.device)
    pruned = source.index_select(dim, index).contiguous()
    if isinstance(tensor, torch.nn.Parameter):
        setattr(
            module,
            attr_name,
            torch.nn.Parameter(pruned, requires_grad=tensor.requires_grad),
        )
    else:
        setattr(module, attr_name, pruned)


def _save_pruned_model(model, pruned_model_dir: pathlib.Path, model_attrs: dict[str, Any]) -> None:
    save_kwargs: dict[str, Any] = {}
    if model_attrs["fused"]:
        # Keep fused expert tensors in the model's native state-dict namespace. The
        # legacy reverse mapping expands them into experts.0.* keys, which cannot
        # be resolved by HF's offloaded-parameter loader for fused expert modules.
        save_kwargs["save_original_format"] = False
    model.save_pretrained(pruned_model_dir, **save_kwargs)


def prune(
    observer_data,
    model,
    prune_args,
    n_experts_to_prune,
    pruned_model_dir,
):
    """
    Prune the model based on the observer data and clustering.
    """
    model_attrs = MODEL_ATTRS[model.__class__.__name__]

    for layer in observer_data:
        if "expert_proba" not in observer_data[layer]:
            # Calculate expert probabilities if not already present
            observer_data[layer]["expert_proba"] = (
                observer_data[layer]["expert_frequency"]
                / observer_data[layer]["total_tokens"]
            )

    if prune_args.perserve_super_experts or prune_args.perserve_outliers:
        super_expert_idx = get_super_expert_indices(observer_data, include_last_layers=prune_args.perserve_outliers)
        metrics = [
            "expert_proba",
            "ean_sum",
            "ean_mean",
            "weighted_expert_frequency_sum",
            "weighted_ean_sum",
            "reap",
            "reap_l2",
            "weighted_ean_sum_l2",
        ]
        for layer in observer_data:
            super_experts_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer][:, 1]
            if len(super_experts_in_layer) > 0:
                for metric in metrics:
                    if metric in observer_data[layer]:
                        observer_data[layer][metric][super_experts_in_layer] = float("inf")

    for layer in tqdm(observer_data, "Pruning layers..."):
        num_experts = observer_data[layer]["expert_frequency"].shape[0]
        if prune_args.prune_method == "ean_ca":
            ean = torch.zeros(
                num_experts, device=get_model_device(model), dtype=torch.float32
            )
            for i in range(num_experts):
                ean[i] = torch.linalg.norm(
                    observer_data[layer]["routed_characteristic_activation"][i], dim=-1
                ).sum()
            _, experts_to_prune = torch.topk(ean, n_experts_to_prune, largest=False)
        else:
            prune_method = prune_args.prune_method
            if prune_method == "frequency":
                prune_method = "expert_frequency"
            saliency_data = observer_data[layer].get(prune_method)
            if saliency_data is None:
                raise ValueError(
                    f"Prune method {prune_args.prune_method} not found in observer data for layer {layer}. "
                    f"Available keys: {list(observer_data[layer].keys())}"
                )
            _, experts_to_prune = torch.topk(
                saliency_data, n_experts_to_prune, largest=False
            )

        retained_expert_indicies = [
            i for i in range(num_experts) if i not in experts_to_prune
        ]
        # prune experts
        moe = get_moe(model, layer)
        if not model_attrs["fused"]:
            all_experts = getattr(moe, model_attrs["experts"])
            retained_experts = [all_experts[i] for i in retained_expert_indicies]
            retained_experts = torch.nn.ModuleList(retained_experts)
            setattr(moe, model_attrs["experts"], retained_experts)
            if model.__class__.__name__.lower() == "Ernie4_5_MoEForCausalLM".lower():
                # transformers version >=4.54
                # prune expert score correction bias too
                moe.moe_statics.e_score_correction_bias.data = (
                    moe.moe_statics.e_score_correction_bias.data[
                        :, retained_expert_indicies
                    ]
                )

            # prune router
            router = getattr(moe, model_attrs["router"])
            router.weight.data = router.weight.data[retained_expert_indicies, :]
            if getattr(router, "bias", None):
                router.bias.data = router.bias.data[retained_expert_indicies]
            router.out_features = len(retained_expert_indicies)
            if hasattr(router, "e_score_correction_bias"):
                router.e_score_correction_bias.data = (
                    router.e_score_correction_bias.data[retained_expert_indicies]
                )
            setattr(moe, model_attrs["router"], router)
        else:
            # prune fused experts (Llama-4, GlmMoeDsa, etc.)
            experts_name = _module_name_for_prune(model, moe.experts)
            _index_select_tensor_attr_for_prune(
                model,
                moe.experts,
                experts_name,
                "gate_up_proj",
                retained_expert_indicies,
            )
            _index_select_tensor_attr_for_prune(
                model,
                moe.experts,
                experts_name,
                "down_proj",
                retained_expert_indicies,
            )
            retained_count = len(retained_expert_indicies)
            _set_fused_expert_counts(moe, model_attrs, retained_count)
            router_attr = model_attrs["router"]
            if hasattr(moe, router_attr):
                router = getattr(moe, router_attr)
            elif hasattr(moe, "router"):
                router = moe.router
            elif hasattr(moe, "gate"):
                router = moe.gate
            else:
                raise AttributeError(
                    f"No router/gate on {moe.__class__.__name__} (expected '{router_attr}')"
                )
            router_name = _module_name_for_prune(model, router)
            _index_select_tensor_attr_for_prune(
                model,
                router,
                router_name,
                "weight",
                retained_expert_indicies,
            )
            _set_fused_router_counts(router, model_attrs, retained_count)
            if hasattr(router, "e_score_correction_bias"):
                _index_select_tensor_attr_for_prune(
                    model,
                    router,
                    router_name,
                    "e_score_correction_bias",
                    retained_expert_indicies,
                )

    # patch config and dump
    logger.info("Saving pruned model...")
    retained_experts = len(retained_expert_indicies)
    setattr(model.config, model_attrs["num_experts"], retained_experts)
    if model.__class__.__name__ == "Ernie4_5_MoeForCausalLM":  # remote-code verson
        model.config.moe_capacity = [
            retained_experts,
            retained_experts,
            retained_experts,
        ]

    pruned_model_dir.mkdir(parents=True, exist_ok=True)
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and getattr(gen_cfg, "top_p", None) is not None:
        if not getattr(gen_cfg, "do_sample", False):
            gen_cfg.do_sample = True
    start = time.time()
    _save_pruned_model(model, pruned_model_dir, model_attrs)
    end = time.time()
    logger.info(
        f"Pruned model saved to {pruned_model_dir} in {end - start:.2f} seconds"
    )
    return pruned_model_dir


def get_pruned_model_dir(
    results_dir: pathlib.Path,
    n_experts_to_prune: int,
    total_experts: int,
    prune_args: PruneArgs,
    seed: int,
    renorm: bool,
    name_prefix: str = None,
) -> pathlib.Path:
    """Generate output directory path for pruned model."""
    compression_ratio_str = f"{(n_experts_to_prune / total_experts):.2f}"
    name_prefix = "" if name_prefix is None else name_prefix
    pruned_model_name = f"{name_prefix}{prune_args.prune_method}"

    if prune_args.perserve_super_experts:
        pruned_model_name += "-perserve_super"
    elif prune_args.perserve_outliers:
        pruned_model_name += "-perserve_outlier"
    if renorm:
        pruned_model_name += f"-renorm_{str(renorm).lower()}"
    pruned_model_name += f"-seed_{seed}"
    pruned_model_name += f"-{compression_ratio_str}"

    pruned_model_dir = results_dir / "pruned_models" / pruned_model_name
    logger.info(f"Using seed {seed}, pruned model dir: {pruned_model_dir}")

    return pruned_model_dir


def main():
    parser = HfArgumentParser(
        (
            ReapArgs,
            DatasetArgs,
            ObserverArgs,
            ModelArgs,
            EvalArgs,
            PruneArgs,
            ClusterArgs,
        )
    )
    reap_args, ds_args, obs_args, model_args, eval_args, prune_args, cluster_args = (
        parser.parse_args_into_dataclasses()
    )
    if prune_args.perserve_super_experts and prune_args.perserve_outliers:
        raise ValueError("Only one of perserve_super_experts or perserve_outliers can be set to True.")
    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    # get local patched model if req'd
    model_name = patched_model_map(model_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = load_causal_lm_for_prune(model_name)
    # record activations or load previously recorded activations
    logger.info(
        f"Running observer to collect activation data for model {model_args.model_name} on dataset {ds_args.dataset_name}."
    )
    observer_data = record_activations(
        model,
        tokenizer,
        reap_args,
        model_args,
        ds_args,
        obs_args,
        results_dir,
    )
    if reap_args.run_observer_only:
        logger.info(
            "Observer run completed. Exiting after collecting activation data since "
            "`run_observer_only` is set to True."
        )
        return

    # pruning
    logger.info("Start of pruning")
    n_experts_to_prune = prune_args.n_experts_to_prune
    if n_experts_to_prune is None:
        if cluster_args.compression_ratio is None:
            raise ValueError(
                "Either n_experts_to_prune or compression_ratio must be set for pruning."
            )
        else:
            # Calculate n_experts_to_prune from compression_ratio
            total_experts = len(
                observer_data[next(iter(observer_data))]["expert_frequency"]
            )
            n_experts_to_prune = int(total_experts * cluster_args.compression_ratio)
            logger.info(
                f"Calculated n_experts to prune: {n_experts_to_prune} from compression_ratio: {cluster_args.compression_ratio}"
            )

    pruned_model_dir = get_pruned_model_dir(
        results_dir, n_experts_to_prune, total_experts, prune_args, reap_args.seed, obs_args.renormalize_router_weights
    )
    if (
        pruned_model_dir.exists()
        and list(pruned_model_dir.glob("*.safetensors"))
        and not prune_args.overwrite_pruned_model
    ):
        logger.info(
            f"Pruned model directory {pruned_model_dir} already exists and contains pruned model files. "
            "Skipping pruning step."
        )
    else:
        logger.info(f"Pruning model to {total_experts - n_experts_to_prune} experts...")
        prune(
            observer_data,
            model,
            prune_args,
            n_experts_to_prune,
            pruned_model_dir,
        )
        logger.info("pruning completed.")

        # smoke test
        if reap_args.smoke_test:
            logger.info("Running smoke test on the merged model...")
            try:
                smoke_test(model, tokenizer)
            except Exception as e:
                logger.error(f"Smoke test failed: {e}")
                pass

        tokenizer.save_pretrained(pruned_model_dir)
        if model_name == "artifacts/models/GLM-4.5-Air":
            # move modelling file
            source_file = pathlib.Path(model_name) / "modeling_glm4_moe.py"
            target_file = pruned_model_dir / "modeling_glm4_moe.py"
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied modeling_glm4_moe.py to {pruned_model_dir}")
            else:
                raise RuntimeError(
                    f"Source file {source_file} does not exist. Cannot copy to {target_file}."
                )

        logger.info("Pruning completed.")

        dump_args_to_yaml(
            pruned_model_dir,
            reap_args=reap_args,
            ds_args=ds_args,
            obs_args=obs_args,
            model_args=model_args,
            eval_args=eval_args,
            prune_args=prune_args,
            cluster_args=cluster_args,
        )

    # eval
    if reap_args.do_eval:
        remove_hook_from_module(model, recurse=True)
        model.to("cpu")
        del model
        del observer_data
        torch.cuda.empty_cache()
        gc.collect()
        model_args.model_name = pruned_model_dir
        run_evaluate(model_args, pruned_model_dir / "eval", eval_args, reap_args.seed)


if __name__ == "__main__":
    main()
