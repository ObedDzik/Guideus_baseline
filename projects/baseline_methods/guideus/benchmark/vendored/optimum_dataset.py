"""VENDORED, do not edit. OPTIMUM-family (OL, UA, PU) single-frame biopsy-core dataset.

Copied from the reference loader code that produced every row of the lab benchmark tables:
    optimum_dataset.py  (source sha256 7d4b36934a2a1d02)
Executable statements are identical to the source (checked by AST comparison when this copy was made);
only comments quoting an example patient record were removed. Changing anything here changes the
benchmark inputs and makes the exported predictions incomparable with the other rows.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset


class OptimumNeedleCoreDataset(Dataset):
    """One last-frame-before-fire image per annotated OPTIMUM biopsy core."""

    def __init__(
        self,
        root_dir: str,
        case_ids: Iterable[str],
        transform=None,
        needle_mask_fname: str = "needle_mask.png",
        prostate_mask_root: str | None = None,
        require_prostate_mask: bool = False,
        require_nonempty_masks: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.prostate_mask_root = (
            Path(prostate_mask_root) if prostate_mask_root else None
        )
        self.require_prostate_mask = bool(require_prostate_mask)
        self.require_nonempty_masks = bool(require_nonempty_masks)
        self.case_ids = tuple(sorted(str(value) for value in case_ids))
        self.transform = transform
        self.data: list[dict] = []
        allowed = set(self.case_ids)

        for case_dir in sorted(self.root_dir.iterdir()):
            if not case_dir.is_dir() or case_dir.name not in allowed:
                continue
            for cine_dir in sorted(case_dir.iterdir()):
                if not cine_dir.is_dir():
                    continue
                image_path = cine_dir / "frames" / "0000.png"
                needle_path = cine_dir / needle_mask_fname
                info_path = cine_dir / "info.json"
                if not (image_path.is_file() and needle_path.is_file() and info_path.is_file()):
                    continue
                prostate_mask_path = cine_dir / "micro_seg_net_prostate_mask.png"
                if (
                    not prostate_mask_path.is_file()
                    and self.prostate_mask_root is not None
                ):
                    auxiliary_path = (
                        self.prostate_mask_root
                        / case_dir.name
                        / cine_dir.name
                        / "micro_seg_net_prostate_mask.png"
                    )
                    if auxiliary_path.is_file():
                        prostate_mask_path = auxiliary_path
                if self.require_prostate_mask and not prostate_mask_path.is_file():
                    continue
                if self.require_nonempty_masks:
                    # A present but empty segmentation is not a usable weak-label
                    # region.  Reject it here instead of silently treating the
                    # complete image as prostate/needle tissue downstream.
                    if Image.open(needle_path).convert("L").getbbox() is None:
                        continue
                    if (
                        not prostate_mask_path.is_file()
                        or Image.open(prostate_mask_path).convert("L").getbbox() is None
                    ):
                        continue
                info = json.loads(info_path.read_text())
                self.data.append(
                    {
                        "image_path": image_path,
                        "needle_mask_path": needle_path,
                        "prostate_mask_path": prostate_mask_path,
                        "info": info,
                    }
                )

        if not self.data:
            raise RuntimeError(
                "No OPTIMUM cores were found under "
                f"{self.root_dir}. Expected CASE/CINE/frames/0000.png, "
                f"{needle_mask_fname}, and info.json."
            )

        self.core_ids = [str(item["info"]["cine_id"]) for item in self.data]
        self.patient_ids = [str(item["info"]["case"]) for item in self.data]
        self.centers = [str(item["info"].get("center", "Unknown")) for item in self.data]
        self.grade_groups = [self._grade(item["info"].get("GG")) for item in self.data]

    @staticmethod
    def _grade(value) -> int:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0
        return int(value)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        out = {
            "image": Image.open(sample["image_path"]).convert("RGB"),
            "needle_mask": Image.open(sample["needle_mask_path"]).convert("L"),
            "path": str(sample["image_path"]),
            "info": dict(sample["info"]),
        }
        prostate_path = sample["prostate_mask_path"]
        if prostate_path.is_file():
            out["prostate_mask"] = Image.open(prostate_path).convert("L")
        if self.transform is not None:
            out = self.transform(out)
        return out

    def list_indices_by_patient_ids(self) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for index, patient_id in enumerate(self.patient_ids):
            result.setdefault(patient_id, []).append(index)
        return result
