"""Reduced PPCV fidelity and efficiency experiments on one H200.

This implementation records the full intervention trace: the initial trajectory,
per-token paraphrase discrepancy scores, selected critical position, candidate
tokens, every continuation, similarity weights, selected answer, and SC baseline.
It is explicitly a reduced Qwen3-8B reproduction, not a claim of exact parity
with the preprint's Qwen3-32B result.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from typing import Any

from src.common import (
    StopController,
    append_jsonl,
    batched,
    immutable_revision,
    load_jsonl,
    object_fingerprint,
    read_yaml,
    resolve_path,
    utc_now,
    write_progress,
)
from src.inference_plan import stable_seed, surface_lookup
from src.scoring import canonical_equivalent, extract_raw_answer, normalize_answer


def build_prompt(tokenizer: Any, system: str, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def pad_sequences(sequences: list[Any], pad_id: int, device: Any):
    import torch

    width = max(sequence.numel() for sequence in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, -sequence.numel() :] = sequence.to(device)
        mask[index, -sequence.numel() :] = 1
    return ids, mask


def generate_batch(
    model: Any,
    tokenizer: Any,
    prefixes: list[Any],
    *,
    max_new_tokens: int,
    seed: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    results = []
    for batch_index, batch in enumerate(batched(prefixes, batch_size)):
        torch.manual_seed(seed + batch_index)
        torch.cuda.manual_seed_all(seed + batch_index)
        ids, mask = pad_sequences(batch, tokenizer.pad_token_id, model.device)
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - started
        input_width = ids.shape[1]
        for row in generated:
            new_ids = row[input_width:].detach().cpu()
            results.append(
                {
                    "new_ids": new_ids,
                    "text": tokenizer.decode(new_ids, skip_special_tokens=True),
                    "output_tokens": int(new_ids.numel()),
                    "latency_seconds": elapsed / len(batch),
                }
            )
    return results


def teacher_force_scores(
    model: Any,
    prompt_ids: Any,
    response_ids: Any,
    top_count: int,
    retain_top_tokens: bool,
) -> tuple[list[float], list[list[int]] | None]:
    import torch

    full = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(model.device)
    with torch.inference_mode():
        logits = model(full, use_cache=False).logits[0]
        start = prompt_ids.numel() - 1
        response_logits = logits[start : start + response_ids.numel()].float()
        log_norm = torch.logsumexp(response_logits, dim=-1)
        actual_logits = response_logits.gather(
            1, response_ids.to(model.device).unsqueeze(1)
        ).squeeze(1)
        top_values, top_ids = torch.topk(response_logits, k=top_count, dim=-1)
        top_prob = torch.exp(top_values[:, 0] - log_norm)
        actual_prob = torch.exp(actual_logits - log_norm)
        discrepancy = (top_prob - actual_prob).detach().cpu().tolist()
        tokens = top_ids.detach().cpu().tolist() if retain_top_tokens else None
    del full, logits, response_logits, log_norm, actual_logits, top_values, top_ids
    torch.cuda.empty_cache()
    return discrepancy, tokens


def select_consistent_rollout(
    rollouts: list[dict[str, Any]], form_weights: dict[str, float]
) -> tuple[dict[str, Any], dict[str, float]]:
    scores: dict[str, float] = {}
    for index, candidate in enumerate(rollouts):
        answer = candidate["parsed_answer"]
        score = 0.0
        if answer is not None:
            for other in rollouts:
                if other["parsed_answer"] == answer:
                    score += form_weights[other["form_id"]]
        scores[str(index)] = score
    best_index = min(
        range(len(rollouts)),
        key=lambda index: (-scores[str(index)], str(rollouts[index]["parsed_answer"]), index),
    )
    return rollouts[best_index], scores


def form_similarity_weights(form_rows: list[dict[str, Any]], encoder: Any) -> dict[str, float]:
    embeddings = encoder.encode(
        [row["prompt_text"] for row in form_rows],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    original = embeddings[0]
    weights = {
        row["form_id"]: max(0.0, float(original @ embedding))
        for row, embedding in zip(form_rows, embeddings, strict=True)
    }
    weights[form_rows[0]["form_id"]] = 1.0
    return weights


def select_items(base: dict, spec: dict) -> list[list[dict[str, Any]]]:
    surfaces = surface_lookup(base, spec["split"])
    originals = [
        row for key, row in surfaces.items() if key[0] == "gsm_symbolic" and key[2] == "original"
    ]
    originals.sort(key=lambda row: row["item_id"])
    random.Random(int(spec["selection_seed"])).shuffle(originals)
    items = []
    for original in originals[: int(spec["item_count"])]:
        items.append(
            [
                surfaces[(original["dataset"], original["item_id"], form)]
                for form in spec["form_ids"]
            ]
        )
    if len(items) != int(spec["item_count"]):
        raise ValueError(
            f"PPCV requested {spec['item_count']} items but only {len(items)} are complete"
        )
    return items


def run_item(
    form_rows: list[dict[str, Any]],
    variant: str,
    spec: dict,
    ppcv: dict,
    model_config: dict,
    tokenizer: Any,
    model: Any,
    encoder: Any,
) -> dict[str, Any]:
    import torch

    item_id = form_rows[0]["item_id"]
    system = model_config["prompt"]["system"]
    prompts = [build_prompt(tokenizer, system, row["prompt_text"]) for row in form_rows]
    prompt_ids = [
        tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        for prompt in prompts
    ]
    max_new_tokens = int(model_config["max_new_tokens"]["non_thinking"]["gsm_symbolic"])
    started = time.monotonic()
    initial = generate_batch(
        model,
        tokenizer,
        [prompt_ids[0]],
        max_new_tokens=max_new_tokens,
        seed=stable_seed(item_id, variant, "initial"),
        batch_size=1,
    )[0]
    response_ids = initial["new_ids"]
    if response_ids.numel() < 2:
        raise RuntimeError(f"Initial trajectory for {item_id} is too short for PPCV")
    discrepancy_by_form = []
    original_top_tokens = None
    candidate_count = int(spec["candidate_count_including_original"])
    for index, form_prompt_ids in enumerate(prompt_ids):
        discrepancy, top_tokens = teacher_force_scores(
            model,
            form_prompt_ids,
            response_ids,
            top_count=max(candidate_count + 1, 2),
            retain_top_tokens=index == 0,
        )
        discrepancy_by_form.append(discrepancy)
        if top_tokens is not None:
            original_top_tokens = top_tokens
    position_scores = [
        max(scores[position] for scores in discrepancy_by_form)
        for position in range(response_ids.numel())
    ]
    special_ids = set(tokenizer.all_special_ids)
    valid_positions = [
        position for position, token in enumerate(response_ids) if int(token) not in special_ids
    ]
    if not valid_positions:
        raise RuntimeError(f"Initial trajectory for {item_id} contains no intervenable tokens")
    critical_position = max(valid_positions, key=position_scores.__getitem__)
    assert original_top_tokens is not None
    actual_token = int(response_ids[critical_position])
    candidate_tokens = [actual_token]
    for token in original_top_tokens[critical_position]:
        if int(token) not in candidate_tokens:
            candidate_tokens.append(int(token))
        if len(candidate_tokens) == candidate_count:
            break

    gold = str(form_rows[0]["canonical_answer"])
    initial_extracted, initial_extraction_method = extract_raw_answer(
        initial["text"], "gsm_symbolic"
    )
    initial_correctness, initial_parser_status = canonical_equivalent(
        gold, initial["text"], initial_extracted
    )
    probe_elapsed_seconds = time.monotonic() - started

    prefixes = []
    metadata = []
    for candidate_index, token in enumerate(candidate_tokens):
        for form_index, form_prompt_ids in enumerate(prompt_ids):
            prefix = torch.cat(
                [
                    form_prompt_ids,
                    response_ids[:critical_position],
                    torch.tensor([token], dtype=torch.long),
                ]
            )
            prefixes.append(prefix)
            metadata.append((candidate_index, token, form_index))
    generations = generate_batch(
        model,
        tokenizer,
        prefixes,
        max_new_tokens=max_new_tokens,
        seed=stable_seed(item_id, variant, "rollouts"),
        batch_size=int(ppcv["teacher_forcing"]["rollout_batch_size"]),
    )
    rollouts = []
    for generation, (candidate_index, token, form_index) in zip(generations, metadata, strict=True):
        full_response_ids = torch.cat(
            [response_ids[:critical_position], torch.tensor([token]), generation["new_ids"]]
        )
        text = tokenizer.decode(full_response_ids, skip_special_tokens=True)
        extracted, extraction_method = extract_raw_answer(text, "gsm_symbolic")
        correctness, parser_status = canonical_equivalent(gold, text, extracted)
        rollouts.append(
            {
                "candidate_index": candidate_index,
                "candidate_token_id": token,
                "candidate_token": tokenizer.decode([token]),
                "form_id": form_rows[form_index]["form_id"],
                "raw_output": text,
                "parsed_answer": normalize_answer(extracted),
                "correctness": correctness,
                "parser_status": parser_status,
                "extraction_method": extraction_method,
                "output_tokens": int(full_response_ids.numel()),
                "latency_seconds": generation["latency_seconds"],
            }
        )
    weights = form_similarity_weights(form_rows, encoder)
    selected, consistency_scores = select_consistent_rollout(rollouts, weights)
    ppcv_elapsed_seconds = time.monotonic() - started

    sc_count = int(spec.get("sc_baseline_samples", 0))
    sc_rows = []
    if sc_count:
        sc_generations = generate_batch(
            model,
            tokenizer,
            [prompt_ids[0]] * sc_count,
            max_new_tokens=max_new_tokens,
            seed=stable_seed(item_id, variant, "sc"),
            batch_size=int(ppcv["teacher_forcing"]["rollout_batch_size"]),
        )
        for generation in sc_generations:
            extracted, extraction_method = extract_raw_answer(generation["text"], "gsm_symbolic")
            correctness, parser_status = canonical_equivalent(gold, generation["text"], extracted)
            sc_rows.append(
                {
                    "raw_output": generation["text"],
                    "parsed_answer": normalize_answer(extracted),
                    "correctness": correctness,
                    "parser_status": parser_status,
                    "extraction_method": extraction_method,
                    "output_tokens": generation["output_tokens"],
                    "latency_seconds": generation["latency_seconds"],
                }
            )
    sc_majority = None
    sc_correctness = None
    if sc_rows:
        counts = Counter(
            row["parsed_answer"] for row in sc_rows if row["parsed_answer"] is not None
        )
        sc_majority = (
            sorted(counts, key=lambda answer: (-counts[answer], str(answer)))[0] if counts else None
        )
        matches = [row["correctness"] for row in sc_rows if row["parsed_answer"] == sc_majority]
        decided = [bool(value) for value in matches if value is not None]
        sc_correctness = (sum(decided) >= len(decided) / 2) if decided else None
    return {
        "run_id": f"ppcv_{variant}",
        "variant": variant,
        "dataset": "gsm_symbolic",
        "item_id": item_id,
        "split": spec["split"],
        "target_model": model_config["model"]["id"],
        "mode": "non_thinking",
        "method": f"ppcv_{variant}",
        "gold_answer": gold,
        "form_ids": [row["form_id"] for row in form_rows],
        "initial_trajectory": initial["text"],
        "initial_answer": normalize_answer(initial_extracted),
        "initial_correctness": initial_correctness,
        "initial_parser_status": initial_parser_status,
        "initial_extraction_method": initial_extraction_method,
        "initial_output_tokens": initial["output_tokens"],
        "probe_elapsed_seconds": probe_elapsed_seconds,
        "critical_position": critical_position,
        "critical_token": tokenizer.decode([actual_token]),
        "critical_score": position_scores[critical_position],
        "position_scores": position_scores,
        "candidate_token_ids": candidate_tokens,
        "form_similarity_weights": weights,
        "rollouts": rollouts,
        "consistency_scores": consistency_scores,
        "selected_answer": selected["parsed_answer"],
        "correctness": selected["correctness"],
        "ppcv_output_tokens": initial["output_tokens"]
        + sum(row["output_tokens"] for row in rollouts),
        "ppcv_elapsed_seconds": ppcv_elapsed_seconds,
        "sc_sample_count": sc_count,
        "sc_majority_answer": sc_majority,
        "sc_correctness": sc_correctness,
        "sc_rollouts": sc_rows,
        "sc_output_tokens": sum(row["output_tokens"] for row in sc_rows),
        "total_output_tokens": initial["output_tokens"]
        + sum(row["output_tokens"] for row in rollouts)
        + sum(row["output_tokens"] for row in sc_rows),
        "elapsed_seconds": time.monotonic() - started,
        "created_at": utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["fidelity", "lite"], required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    base = read_yaml("configs/experiment_main.yaml")
    ppcv = read_yaml("configs/experiment_ppcv.yaml")
    model_config = read_yaml(ppcv["model_config"])
    spec = ppcv[args.variant]
    model_id = model_config["model"]["id"]
    revision = immutable_revision(
        model_config["model"].get("revision"), model_id, "environment/model_revisions.json"
    )
    embedding_id = ppcv["selection"]["embedding_model"]
    embedding_revision = immutable_revision(
        ppcv["selection"].get("embedding_revision"),
        embedding_id,
        "environment/model_revisions.json",
    )
    fingerprint = object_fingerprint(
        {"base": base, "ppcv": ppcv, "model": model_config, "variant": args.variant}
    )
    output = resolve_path(args.output or f"outputs/generations/ppcv_{args.variant}.jsonl")
    progress = resolve_path(f"outputs/checkpoints/ppcv_{args.variant}.json")
    existing_rows = load_jsonl(output)
    if any(row.get("config_fingerprint") not in {None, fingerprint} for row in existing_rows):
        raise RuntimeError("Refusing to resume PPCV output created with a different configuration")
    completed = {row["item_id"] for row in existing_rows}
    items = [rows for rows in select_items(base, spec) if rows[0]["item_id"] not in completed]
    if not items:
        print(f"[ppcv] {args.variant} is already complete")
        return

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoder = SentenceTransformer(embedding_id, revision=embedding_revision, device="cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    stop = StopController()
    stop.install()
    total_expected = len(items) + len(completed)
    for form_rows in items:
        record = run_item(
            form_rows,
            args.variant,
            spec,
            ppcv,
            model_config,
            tokenizer,
            model,
            encoder,
        )
        record["model_revision"] = revision
        record["embedding_revision"] = embedding_revision
        record["config_fingerprint"] = fingerprint
        append_jsonl(output, [record])
        completed.add(record["item_id"])
        write_progress(
            progress,
            run_id=f"ppcv_{args.variant}",
            config_fingerprint=fingerprint,
            completed=len(completed),
            expected=total_expected,
            stop_controller=stop,
            extra={"output_path": str(output)},
        )
        print(f"[ppcv] flushed {len(completed)}/{total_expected}", flush=True)
        if stop.requested:
            return


if __name__ == "__main__":
    main()
