"""Run append-only, signal-aware vLLM inference for main/SCoP/external plans."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.common import (
    StopController,
    append_jsonl,
    batched,
    completed_keys,
    immutable_revision,
    object_fingerprint,
    prompt_hash,
    read_yaml,
    record_key,
    resolve_path,
    utc_now,
    write_progress,
)
from src.inference_plan import external_plan, main_plan, scop_plan


def build_prompt(tokenizer: Any, system_prompt: str, user_prompt: str, mode: str) -> str:
    if mode == "reasoning":
        # DeepSeek-R1 recommends no system role, with all instructions in the
        # user turn, and an explicit opening think tag for reliable reasoning.
        messages = [
            {
                "role": "user",
                "content": (
                    f"{user_prompt}\n\nPlease reason step by step, and put your final "
                    "answer within \\boxed{}."
                ),
            }
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered + "<think>\n"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if mode in {"thinking", "non_thinking"}:
        kwargs["enable_thinking"] = mode == "thinking"
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def sampling_settings(model_config: Mapping[str, Any], mode: str, protocol: str) -> dict[str, Any]:
    if mode == "reasoning":
        # DeepSeek-R1-distill recommended non-greedy defaults; kept explicit.
        return {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "min_p": 0.0}
    return dict(model_config["modes"][mode][protocol])


def max_tokens_for(model_config: Mapping[str, Any], mode: str, dataset: str) -> int:
    if mode == "reasoning":
        return 4096 if dataset == "math_500" else 2048
    return int(model_config["max_new_tokens"][mode][dataset])


def make_plan(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    base = read_yaml(args.config)
    model_config = read_yaml(base["model_config"])
    if args.experiment == "main":
        plan = main_plan(base, args.split, args.mode, args.protocol)
        experiment_config = base
    elif args.experiment == "scop":
        experiment_config = read_yaml("configs/experiment_scop.yaml")
        plan = scop_plan(base, experiment_config, args.split, args.mode, args.protocol)
    else:
        experiment_config = read_yaml("configs/experiment_external.yaml")
        plan = external_plan(base, experiment_config, args.mode, args.protocol)
    fingerprint = object_fingerprint(
        {"base": base, "model": model_config, "experiment": experiment_config, "args": vars(args)}
    )
    return base, model_config, plan, fingerprint


def validate_existing(progress_path: Path, fingerprint: str) -> None:
    if not progress_path.exists():
        return
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("config_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Refusing to resume {progress_path}: its config fingerprint differs from this command"
        )


def run(args: argparse.Namespace) -> None:
    base, model_config, plan, fingerprint = make_plan(args)
    if args.mode == "controlled" or args.protocol == "controlled":
        if args.split != "pilot":
            raise ValueError(
                "The controlled decoding ablation is preregistered for the pilot split only"
            )
    model_key = "external_model" if args.experiment == "external" else "model"
    model_spec = model_config[model_key]
    revision = immutable_revision(
        model_spec.get("revision"), model_spec["id"], "environment/model_revisions.json"
    )
    run_id = args.run_id or f"{args.experiment}_{args.split}_{args.mode}_{args.protocol}"
    output_path = resolve_path(args.output or f"outputs/generations/{run_id}.jsonl")
    progress_path = resolve_path(f"outputs/checkpoints/{run_id}.json")
    validate_existing(progress_path, fingerprint)
    seen, duplicates = completed_keys(output_path)
    if duplicates:
        raise RuntimeError(f"Existing output contains {len(duplicates)} duplicate result keys")
    pending = [row for row in plan if record_key(row) not in seen]
    print(f"[run] {len(seen)}/{len(plan)} records already complete; {len(pending)} pending")
    if not pending:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["id"], revision=revision, trust_remote_code=bool(model_spec["trust_remote_code"])
    )
    llm = LLM(
        model=model_spec["id"],
        revision=revision,
        dtype=model_spec["dtype"],
        trust_remote_code=bool(model_spec["trust_remote_code"]),
        gpu_memory_utilization=float(model_spec["gpu_memory_utilization"]),
        max_model_len=int(model_spec["max_model_len"]),
        enforce_eager=bool(model_spec.get("enforce_eager", False)),
        generation_config="vllm",
    )
    stop = StopController()
    stop.install()
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pending:
        grouped[max_tokens_for(model_config, args.mode, row["dataset"])].append(row)
    settings = sampling_settings(model_config, args.mode, args.protocol)
    completed = len(seen)
    started = time.monotonic()
    for max_tokens, rows in sorted(grouped.items()):
        for batch in batched(rows, int(base["inference_batch_size"])):
            prompts = [
                build_prompt(
                    tokenizer, model_config["prompt"]["system"], row["prompt_text"], args.mode
                )
                for row in batch
            ]
            params = [
                SamplingParams(
                    n=1,
                    temperature=float(settings["temperature"]),
                    top_p=float(settings["top_p"]),
                    top_k=int(settings["top_k"]),
                    min_p=float(settings.get("min_p", 0.0)),
                    max_tokens=max_tokens,
                    seed=int(row["seed"]),
                    logprobs=1,
                )
                for row in batch
            ]
            batch_start = time.monotonic()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            batch_seconds = time.monotonic() - batch_start
            try:
                import torch

                peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
            except Exception:  # noqa: BLE001
                peak_gib = None
            records = []
            for row, prompt, output in zip(batch, prompts, outputs, strict=True):
                completion = output.outputs[0]
                output_tokens = len(completion.token_ids)
                records.append(
                    {
                        "run_id": run_id,
                        "dataset": row["dataset"],
                        "item_id": row["item_id"],
                        "split": row["split"],
                        "form_id": row["form_id"],
                        "transformation_type": row["transformation_type"],
                        "paraphrase_generator": row.get("paraphrase_generator"),
                        "target_model": model_spec["id"],
                        "model_revision": revision,
                        "mode": args.mode,
                        "protocol": args.protocol,
                        "seed": int(row["seed"]),
                        "method": row["method"],
                        "sample_index": row.get("sample_index"),
                        "sampling_config": {**settings, "max_new_tokens": max_tokens},
                        "prompt_hash": prompt_hash(prompt),
                        "config_fingerprint": fingerprint,
                        "gold_answer": row["gold_answer"],
                        "canonical_answer": row["canonical_answer"],
                        "raw_output": completion.text,
                        "parsed_answer": None,
                        "correctness": None,
                        "input_tokens": len(output.prompt_token_ids),
                        "output_tokens": output_tokens,
                        "normalized_log_probability": (
                            float(completion.cumulative_logprob) / max(1, output_tokens)
                            if completion.cumulative_logprob is not None
                            else None
                        ),
                        "latency_seconds": batch_seconds / len(batch),
                        "batch_latency_seconds": batch_seconds,
                        "gpu_seconds": batch_seconds / len(batch),
                        "peak_gpu_memory_gib": peak_gib,
                        "truncated": completion.finish_reason == "length",
                        "finish_reason": completion.finish_reason,
                        "parser_status": "not_scored",
                        "judge_status": "not_requested",
                        "created_at": utc_now(),
                    }
                )
            append_jsonl(output_path, records)
            completed += len(records)
            write_progress(
                progress_path,
                run_id=run_id,
                config_fingerprint=fingerprint,
                completed=completed,
                expected=len(plan),
                stop_controller=stop,
                extra={
                    "output_path": str(output_path),
                    "elapsed_seconds": time.monotonic() - started,
                    "last_batch_seconds": batch_seconds,
                },
            )
            print(f"[run] flushed {completed}/{len(plan)}", flush=True)
            if stop.requested:
                print("[run] safe stop complete; resubmit the same command to resume", flush=True)
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_main.yaml")
    parser.add_argument("--experiment", choices=["main", "scop", "external"], default="main")
    parser.add_argument("--split", choices=["smoke", "pilot", "confirmatory"], required=True)
    parser.add_argument("--mode", choices=["thinking", "non_thinking", "reasoning"], required=True)
    parser.add_argument("--protocol", choices=["practical", "controlled"], default="practical")
    parser.add_argument("--run-id")
    parser.add_argument("--output")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
