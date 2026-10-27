"""VENDORED, do not edit. OPTIMUM-family item adapter (probe_top vertical flip, NaN-preserving clinical values).

Copied from the reference loader code that produced every row of the lab benchmark tables:
    loaders.py (class _ProstNFoundDatasetAdapterOptimum)  (source sha256 57c88b407d1b8857)
Executable statements are identical to the source (checked by AST comparison when this copy was made);
only comments quoting an example patient record were removed. Changing anything here changes the
benchmark inputs and makes the exported predictions incomparable with the other rows.
"""
import numpy as np


class _ProstNFoundDatasetAdapterOptimum:
    def __init__(self, image_orientation="probe_top"):
        self.image_orientation = image_orientation

    def __call__(self, item):
        """
        Adapter function to convert the Optimum dataset items to the format expected by prostnfound.
        """

        bmode = item["image"]
        bmode = np.array(bmode)[..., 0]
        needle_mask = np.array(item["needle_mask"])
        # The current export has expert/MicroSegNet prostate masks for OL but
        # not for every UA core.  Preserve a real mask whenever it exists and
        # use the full image only for the explicitly missing-mask cases, as in
        # the supplied OPTIMUM example loader.
        prostate_mask = np.array(
            item.get("prostate_mask", np.ones_like(needle_mask))
        )

        if self.image_orientation == "probe_top":
            bmode = np.flipud(bmode)  # Flip the image vertically
            needle_mask = np.flipud(needle_mask)  # Flip the mask vertically
            prostate_mask = np.flipud(prostate_mask)  # Flip the mask vertically

        info = item["info"]

        # 'isFusionVu': True,

        grade_group = info.get("GG", 0)
        if grade_group is None or (
            isinstance(grade_group, float) and np.isnan(grade_group)
        ):
            grade_group = 0
        grade_group = int(grade_group)
        involvement = info.get("% Cancer", 0.0)
        involvement_missing = involvement is None or (
            isinstance(involvement, float) and np.isnan(involvement)
        )
        # Missing percentage is a valid q=0 target only for pathology-benign
        # cores.  OPTIMUM also contains confirmed GG1--GG4 carcinomas for
        # which the pathologist did not report a percentage; retain their
        # classification/grade labels but mask them out of q-dependent losses.
        involvement_valid = not involvement_missing or grade_group == 0
        if involvement_missing:
            involvement = 0.0

        return {
            "bmode": bmode,
            "needle_mask": needle_mask,
            "prostate_mask": prostate_mask,
            "grade": info.get("Diagnosis", "Unknown"),
            "pct_cancer": float(involvement),
            "involvement_valid": bool(involvement_valid),
            "psa": info.get("psa", np.nan),
            "age": info.get("age", np.nan),
            # PSAD is absent from this OPTIMUM export.  NaN deliberately routes
            # through the transform's training-set-mean imputation instead of
            # fabricating a zero-density patient.
            "approx_psa_density": info.get("approx_psa_density", np.nan),
            "family_history": info.get("family_history", np.nan),
            "center": info.get("center", "Unknown"),
            "all_cores_benign": info.get("all_cores_benign", False),
            "core_id": info.get("cine_id", "Unknown"),
            "patient_id": info["case"],
            "loc": info["Sample ID"],
            "grade_group": grade_group,
            "clinically_significant": info.get(
                "clinically_significant", grade_group >= 3
            ),
        }
