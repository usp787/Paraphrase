"""Blindly adjudicate only parser-layer disagreements or undecidable answers."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from src.common import (
    append_jsonl,
    batched,
    immutable_revision,
    load_jsonl,
    read_yaml,
    record_key,
    utc_now,
)

JUDGMENT_PATH = "outputs/checkpoints/score_judgments.jsonl"


def blinded_prompt(gold: str, response: str) -> str:
    return f"""Decide whether the candidate response's final answer is mathematically equivalent
to the reference answer. Ignore prose and derivation quality. You are not told the model,
prompt form, or experimental method. Return exactly one JSON object:
{{"equivalent": true, "uncertain": false, "reason": "brief reason"}}

REFERENCE ANSWER:
{gold}

CANDIDATE RESPONSE:
{response}
"""


def parse(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"parse_ok": False, "uncertain": True, "raw": text}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"parse_ok": False, "uncertain": True, "raw": text}
    ok = isinstance(value.get("equivalent"), bool) and isinstance(value.get("uncertain"), bool)
    return {**value, "parse_ok": ok, "raw": text}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    config = read_yaml("configs/experiment_main.yaml")
    settings = config["paraphrase_generation"]
    existing = {record_key(row) for row in load_jsonl(JUDGMENT_PATH)}
    pending = [
        row
        for row in load_jsonl(args.scores)
        if row.get("judge_required")
        and row.get("judge_status") == "pending"
        and record_key(row) not in existing
    ]
    if not pending:
        print("[judge] no pending score adjudications")
        return
    model_id = settings["judge_model"]
    revision = immutable_revision(
        settings.get("judge_revision"), model_id, "environment/model_revisions.json"
    )
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        revision=revision,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.88,
        generation_config="vllm",
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=192, seed=0)
    for batch in batched(pending, args.batch_size):
        prompts = [blinded_prompt(str(row["canonical_answer"]), row["raw_output"]) for row in batch]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        append_jsonl(
            JUDGMENT_PATH,
            [
                {
                    **{
                        field: row[field]
                        for field in ("dataset", "item_id", "form_id", "mode", "seed", "method")
                    },
                    "judge_model": model_id,
                    "judge_revision": revision,
                    "judgment": parse(output.outputs[0].text),
                    "created_at": utc_now(),
                }
                for row, output in zip(batch, outputs, strict=True)
            ],
        )
        print(f"[judge] flushed {len(batch)} decisions", flush=True)


if __name__ == "__main__":
    main()
