"""Reference image resolution for paired QA runs.

The current benchmark stores test images under data/eval_set/{bug,normal}/...
with matching normal references for bug images. In production, callers can pass
an explicit reference path or point this resolver at a reference mirror.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_reference(
    image_path: str | Path,
    eval_root: str | Path | None = None,
    reference_root: str | Path | None = None,
) -> dict:
    """Return a reference path and status for an image.

    Resolution order:
    1. If the image is a normal eval image, use itself as the known-good image.
    2. If the image is a bug eval image, map bug/.../,bug.png to normal/.../,normal.png.
    3. If reference_root is given, mirror the relative path there.

    Missing or ambiguous references are reported in the status; callers should
    treat non-ok statuses as suspicious rather than falling back to normal.
    """
    image = Path(image_path)
    eval_base = Path(eval_root) if eval_root else _infer_eval_root(image)
    candidates: list[Path] = []

    if eval_base:
        base_label = eval_base.name.lower()
        if base_label == "normal":
            candidates.append(image)
        elif base_label == "bug":
            mapped = eval_base.parent / "normal" / _best_effort_relative(image, eval_base)
            mapped = mapped.with_name(mapped.name.replace(",bug", ",normal"))
            candidates.append(mapped)
        try:
            rel = image.resolve().relative_to(eval_base.resolve())
        except ValueError:
            try:
                rel = image.relative_to(eval_base)
            except ValueError:
                rel = None
        if rel and rel.parts:
            label = rel.parts[0].lower()
            if label == "normal":
                candidates.append(image)
            elif label == "bug":
                mapped_parts = ("normal",) + rel.parts[1:]
                mapped = eval_base.joinpath(*mapped_parts)
                mapped = mapped.with_name(mapped.name.replace(",bug", ",normal"))
                candidates.append(mapped)

    if reference_root:
        ref_base = Path(reference_root)
        rel = _best_effort_relative(image, eval_base)
        if rel is not None:
            candidates.append(ref_base / rel)
        candidates.append(ref_base / image.name)

    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)

    existing = [c for c in unique if c.exists()]
    if len(existing) == 1:
        return {
            "status": "ok",
            "reference_path": str(existing[0]),
            "reason": "reference_resolved",
        }
    if len(existing) > 1:
        return {
            "status": "ambiguous_reference",
            "reference_path": "",
            "reason": ";".join(str(p) for p in existing),
        }
    return {
        "status": "missing_reference",
        "reference_path": "",
        "reason": "no matching reference found",
    }


def _infer_eval_root(image: Path) -> Optional[Path]:
    parts = list(image.parts)
    for idx, part in enumerate(parts):
        if part == "eval_set":
            return Path(*parts[: idx + 1])
    return None


def _best_effort_relative(image: Path, base: Optional[Path]) -> Optional[Path]:
    if base:
        try:
            return image.resolve().relative_to(base.resolve())
        except ValueError:
            try:
                return image.relative_to(base)
            except ValueError:
                pass
    return Path(image.name)
