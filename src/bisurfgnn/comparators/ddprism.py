"""Load inspected DD-PRiSM classes from hash-bound author notebooks.

This adapter changes the pathway count only. Dataset, information and loss
adaptations are recorded by the runner, not hidden inside the author model.
Author code stays in its separately licensed cache (CC-BY-NC-SA-4.0).
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def verify_source(root: Path, key: str, filename: str) -> Path:
    manifest = json.loads((root / "data/raw_manifests/model_comparators.json").read_text())
    entry = next(f for f in manifest["sources"][key]["files"] if f["file"] == filename)
    path = root / "data/raw/model_comparators" / key / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"Author source hash mismatch: {key}/{filename}")
    return path


def class_code(path: Path, classname: str) -> ast.Module:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    found = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if f"class {classname}(" in source:
            found.extend(node for node in ast.parse(source).body
                         if isinstance(node, ast.ClassDef) and node.name == classname)
    if len(found) != 1:
        raise RuntimeError(f"Expected one inspected author class {classname}")
    return ast.Module(body=found, type_ignores=[])


def load_author_classes(root: Path, pathway_count: int):
    if pathway_count < 1:
        raise ValueError("At least one pathway is required")
    namespace = {"nn": nn, "torch": torch,
                 "batch_dot": lambda a, b: torch.sum(a * b, dim=1, keepdim=True)}
    mono = class_code(verify_source(root, "ddprism", "00_MonotherapyUtils.ipynb"),
                      "MonotherapyModel")
    combo = class_code(verify_source(root, "ddprism", "00_CombinationtherapyUtils.ipynb"),
                       "CombinationTherapyModel")
    changed = 0
    for node in ast.walk(combo):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
            and t.value.id == "self" and t.attr == "num_pathway" for t in node.targets
        ):
            if not isinstance(node.value, ast.Constant) or node.value.value != 186:
                raise RuntimeError("Author pathway-count assignment changed")
            node.value = ast.Constant(value=pathway_count)
            changed += 1
    if changed != 1:
        raise RuntimeError("Expected exactly one documented pathway-count adaptation")
    exec(compile(ast.fix_missing_locations(mono), "ddprism_author_mono", "exec"), namespace)
    exec(compile(ast.fix_missing_locations(combo), "ddprism_author_combo", "exec"), namespace)
    return namespace["MonotherapyModel"], namespace["CombinationTherapyModel"]


def capture_attention(model, features):
    captured = []
    handle = model.sample_attention_block.register_forward_hook(
        lambda module, args, output: captured.append(output.detach()))
    try:
        with torch.no_grad():
            model(features)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Pathway attention was not captured exactly once")
    return captured[0]
