"""Resolve Hub artifacts to commits and optionally download them to ``HF_HOME``."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from src.common import atomic_write_json, read_yaml, utc_now


def collect_artifacts(group: str) -> list[tuple[str, str, str | None]]:
    main = read_yaml("configs/experiment_main.yaml")
    models = read_yaml(main["model_config"])
    ppcv = read_yaml("configs/experiment_ppcv.yaml")
    primary = [("model", models["model"]["id"], models["model"].get("revision"))]
    paraphrase = [
        (
            "model",
            main["paraphrase_generation"]["generator_model"],
            main["paraphrase_generation"].get("generator_revision"),
        ),
        (
            "model",
            main["paraphrase_generation"]["judge_model"],
            main["paraphrase_generation"].get("judge_revision"),
        ),
    ]
    optional = [
        ("model", models["external_model"]["id"], models["external_model"].get("revision")),
        (
            "model",
            ppcv["selection"]["embedding_model"],
            ppcv["selection"].get("embedding_revision"),
        ),
    ]
    if group == "primary":
        return primary
    if group == "paraphrase":
        return paraphrase
    if group == "ppcv":
        return primary + optional[-1:]
    return primary + paraphrase + optional


def resolve(kind: str, artifact_id: str, requested: str | None) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    info = (
        api.model_info(artifact_id, revision=requested or "main")
        if kind == "model"
        else api.dataset_info(artifact_id, revision=requested or "main")
    )
    if not info.sha:
        raise RuntimeError(f"Could not resolve {artifact_id} to a Hub commit")
    return info.sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["primary", "paraphrase", "ppcv", "all"], default="all")
    parser.add_argument("--resolve-only", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("HF_HOME"):
        raise OSError("Set HF_HOME to scratch before staging model weights")
    from huggingface_hub import snapshot_download

    revision_path = "environment/model_revisions.json"
    resolved: dict[str, dict[str, Any]] = {}
    if os.path.exists(revision_path):
        with open(revision_path, encoding="utf-8") as handle:
            resolved.update(json.load(handle))
    for kind, artifact_id, requested in collect_artifacts(args.group):
        revision = resolve(kind, artifact_id, requested)
        entry: dict[str, Any] = {
            "kind": kind,
            "revision": revision,
            "resolved_at": utc_now(),
        }
        if not args.resolve_only:
            entry["snapshot_path"] = snapshot_download(repo_id=artifact_id, revision=revision)
        resolved[artifact_id] = entry
        print(f"[stage] {artifact_id}@{revision}", flush=True)
    atomic_write_json(revision_path, resolved)


if __name__ == "__main__":
    main()
