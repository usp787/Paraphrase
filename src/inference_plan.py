"""Build deterministic main, SCoP, and external-validity request plans."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from typing import Any

from src.common import load_jsonl


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def surface_lookup(
    config: Mapping[str, Any], split: str
) -> dict[tuple[str, str, str], dict[str, Any]]:
    manifest = [row for row in load_jsonl(config["manifest_path"]) if row["split"] == split]
    surfaces: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in manifest:
        key = (row["dataset"], row["item_id"], "original")
        surfaces[key] = {
            **row,
            "form_id": "original",
            "transformation_type": "original",
            "prompt_text": row["original_text"],
            "paraphrase_generator": None,
        }
    for row in load_jsonl(config["paraphrases_path"]):
        if row["split"] != split:
            continue
        key = (row["dataset"], row["item_id"], row["form_id"])
        if key in surfaces:
            raise ValueError(f"Duplicate surface record: {key}")
        surfaces[key] = {
            **row,
            "prompt_text": row["paraphrase_text"],
        }
    return surfaces


def _select_surfaces(
    surfaces: Mapping[tuple[str, str, str], dict[str, Any]], form_ids: list[str]
) -> list[dict[str, Any]]:
    originals = [row for key, row in surfaces.items() if key[2] == "original"]
    originals.sort(key=lambda row: (row["dataset"], row["item_id"]))
    selected = []
    for original in originals:
        for form_id in form_ids:
            key = (original["dataset"], original["item_id"], form_id)
            if key not in surfaces:
                raise KeyError(f"Missing required surface {key}")
            selected.append(surfaces[key])
    return selected


def main_plan(
    config: Mapping[str, Any], split: str, mode: str, protocol: str
) -> list[dict[str, Any]]:
    surfaces = surface_lookup(config, split)
    selected = _select_surfaces(surfaces, list(config["main_form_ids"]))
    return [
        {
            **row,
            "mode": mode,
            "protocol": protocol,
            "seed": int(seed),
            "method": "main",
        }
        for row in selected
        for seed in config["sampling_seeds"]
    ]


def scop_plan(
    base: Mapping[str, Any], scop: Mapping[str, Any], split: str, mode: str, protocol: str
) -> list[dict[str, Any]]:
    surfaces = surface_lookup(base, split)
    rows: list[dict[str, Any]] = []
    for method in scop["methods"]:
        selected = _select_surfaces(surfaces, list(method["form_ids"]))
        for row in selected:
            for sample_index in range(int(method["samples_per_form"])):
                rows.append(
                    {
                        **row,
                        "mode": mode,
                        "protocol": protocol,
                        "seed": stable_seed(
                            base["run_seed"],
                            method["name"],
                            row["item_id"],
                            row["form_id"],
                            sample_index,
                        ),
                        "sample_index": sample_index,
                        "method": method["name"],
                    }
                )
    return rows


def external_plan(
    base: Mapping[str, Any], external: Mapping[str, Any], mode: str, protocol: str
) -> list[dict[str, Any]]:
    split = external["split"]
    surfaces = surface_lookup(base, split)
    originals = [row for key, row in surfaces.items() if key[2] == "original"]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in originals:
        by_dataset.setdefault(row["dataset"], []).append(row)
    total = int(external["stratified_item_count"])
    datasets = sorted(by_dataset)
    per_dataset, remainder = divmod(total, len(datasets))
    selected_ids: set[tuple[str, str]] = set()
    for index, dataset in enumerate(datasets):
        rows = sorted(by_dataset[dataset], key=lambda row: row["item_id"])
        random.Random(f"{external['selection_seed']}:{dataset}").shuffle(rows)
        take = per_dataset + (1 if index < remainder else 0)
        selected_ids.update((dataset, row["item_id"]) for row in rows[:take])
    selected_surfaces = [
        row
        for row in _select_surfaces(surfaces, list(external["form_ids"]))
        if (row["dataset"], row["item_id"]) in selected_ids
    ]
    return [
        {
            **row,
            "mode": mode,
            "protocol": protocol,
            "seed": int(seed),
            "method": "external_main",
        }
        for row in selected_surfaces
        for seed in external["sampling_seeds"]
    ]
