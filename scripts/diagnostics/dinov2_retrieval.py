"""
dinov2_retrieval.py  (Task 4 — DINOv2 retrieval pass on a DINOv2-embedded bundle)

Pre-registered in docs/decisions.md (2026-06-12). The §5.1–§5.5 saturation /
selection-invariance result was measured under CLIP ViT-L/14 embeddings. A live
alternative explanation is that the saturation is a CLIP-specific caption/scene
artifact. This script re-runs the *same* saturation + retrieval diagnostics on a
bundle whose frame_embeddings.npz holds DINOv2 (vision-only, self-supervised)
embeddings instead, to ask: does the finding survive a different backbone?

It re-uses the exact CLIP-side machinery so the comparison is apples-to-apples:
  * similarity distributions  -> similarity_distributions.{intra_episode_cosines,
    mean_pair_cosines, summary_row}      => results/tables/dinov2_similarity.{md,csv}
  * Top-1/Top-5 grid + boot CIs and the 40-pair paired permutation grid
    -> stats.{method_correctness, bootstrap_ci, perm_test}
                                          => results/tables/dinov2_retrieval.{md,csv}
                                             results/tables/dinov2_permutation.{md,csv}

Pre-registered decision rule (evaluated here, printed as VERDICT):
  PERSISTS (saturation is data-intrinsic, not a CLIP artifact) iff
    - intra-episode median ~ inter-episode same-task median (within 0.05), AND
    - the inter-task gap stays the only real separation, AND
    - 0/40 method-pair Top-1 permutation tests are significant (selection-invariant).
  BREAKS (saturation is CLIP-specific; DINOv2 recovers selection signal) iff
    >= 1/40 method pairs is significant at p<0.05 OR the intra/inter-task structure
    materially changes. Either way the numbers are reported honestly.

Pure numpy + pandas; reads only the bundle (DINOv2-embedded). No GPU.

Usage (from keyframe-selector/, after exporting the DINOv2 bundle):
    python scripts/export_eval_bundle.py --backbone dinov2 \\
        --out_dir results/bundle_dinov2 --allow_embed
    python scripts/diagnostics/dinov2_retrieval.py \\
        --bundle results/bundle_dinov2 --out_dir results_dinov2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling imports

import numpy as np
import pandas as pd

from bundle import Bundle  # noqa: E402
from similarity_distributions import (  # noqa: E402
    intra_episode_cosines, mean_pair_cosines, summary_row)
from stats import method_correctness, bootstrap_ci, perm_test  # noqa: E402

GATE_INTRA_SAME = 0.05   # |intra median - same-task median| below this => "tight"


def similarity_table(b: Bundle) -> tuple[pd.DataFrame, dict]:
    """The three-distribution saturation table + the medians used by the verdict."""
    intra = intra_episode_cosines(b)
    same_task, diff_task = mean_pair_cosines(b)
    df = pd.DataFrame([
        summary_row("intra_episode_frame_pairs", intra),
        summary_row("inter_episode_same_task_means", same_task),
        summary_row("inter_task_means", diff_task),
    ])
    med = {"intra": float(np.median(intra)) if intra.size else float("nan"),
           "same": float(np.median(same_task)) if same_task.size else float("nan"),
           "inter": float(np.median(diff_task)) if diff_task.size else float("nan")}
    return df, med


def retrieval_grid(b: Bundle, boot: int, rng):
    """(ci_df, perm_df, n_sig): Top-1/5 grid with boot CIs + 40-pair permutation."""
    methods, k_sweep = b.methods, b.k_sweep
    corr = {(m, k): method_correctness(b, m, k) for m in methods for k in k_sweep}

    ci_rows = []
    for m in methods:
        for k in k_sweep:
            t1, lo1, hi1 = bootstrap_ci(corr[(m, k)][1], boot, rng)
            t5, lo5, hi5 = bootstrap_ci(corr[(m, k)][5], boot, rng)
            ci_rows.append({"method": m, "K": k,
                            "top_1": t1, "t1_lo": lo1, "t1_hi": hi1,
                            "top_5": t5, "t5_lo": lo5, "t5_hi": hi5})
    ci_df = pd.DataFrame(ci_rows)

    perm_rows = []
    for k in k_sweep:
        for i, ma in enumerate(methods):
            for mb in methods[i + 1:]:
                obs, p = perm_test(corr[(ma, k)][1], corr[(mb, k)][1], boot, rng)
                perm_rows.append({"K": k, "method_a": ma, "method_b": mb,
                                  "diff_top1": obs, "p_value": p,
                                  "sig_0.05": p < 0.05})
    perm_df = pd.DataFrame(perm_rows)
    n_sig = int(perm_df["sig_0.05"].sum())
    return ci_df, perm_df, n_sig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/bundle_dinov2")
    ap.add_argument("--out_dir", default="results_dinov2")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    b = Bundle(args.bundle)
    if not b.has_indices():
        sys.exit("Bundle has no keyframes.jsonl — re-export without --no_indices.")
    backbone = b.config.get("retrieval_backbone", "unknown")
    emb_model = b.config.get("embedding_model", "unknown")
    D = b.frames(b.episode_indices[0]).shape[1]
    print(f"Loaded bundle: {len(b.query_eps())} queries, {len(b.gallery_eps())} gallery "
          f"| backbone={backbone}  model={emb_model}  dim={D}")
    if backbone == "clip":
        print("  NOTE: this is a CLIP bundle — running here only validates the driver; "
              "the DINOv2 verdict requires a --backbone dinov2 bundle.")

    rng = np.random.default_rng(args.seed)
    sim_df, med = similarity_table(b)
    ci_df, perm_df, n_sig = retrieval_grid(b, args.boot, rng)

    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(out / "tables" / "dinov2_similarity.csv", index=False)
    ci_df.to_csv(out / "tables" / "dinov2_retrieval.csv", index=False)
    perm_df.to_csv(out / "tables" / "dinov2_permutation.csv", index=False)

    # ---- pre-registered verdict ----------------------------------------- #
    tight = abs(med["intra"] - med["same"]) <= GATE_INTRA_SAME
    gap = med["same"] - med["inter"]
    persists = tight and (n_sig == 0)
    verdict = "PERSISTS (data-intrinsic)" if persists else "BREAKS (backbone-specific)"
    note = [
        f"DINOv2 retrieval pass (docs/decisions.md 2026-06-12), backbone={backbone}, "
        f"dim={D}, n_queries={len(b.query_eps())}:",
        f"  intra={med['intra']:.3f}  same-task={med['same']:.3f}  "
        f"inter-task={med['inter']:.3f}  (intra~same within {GATE_INTRA_SAME}: {tight}; "
        f"task gap same-inter = {gap:+.3f})",
        f"  method-pair Top-1 permutation: {n_sig} / {len(perm_df)} significant at p<0.05",
        f"  VERDICT: saturation {verdict}",
    ]
    _write_md(sim_df, out / "tables" / "dinov2_similarity.md",
              banner="> DINOv2 saturation distributions (intra/same-task/inter-task "
                     "cosine). Pre-registered cross-encoder check; see docs/decisions.md "
                     "(2026-06-12).", note=note)
    _write_md(ci_df, out / "tables" / "dinov2_retrieval.md",
              banner="> DINOv2 Top-1/Top-5 grid with bootstrap 95% CIs (method x K). "
                     "Compare against the CLIP retrieval_cis.* grid.")
    _write_md(perm_df, out / "tables" / "dinov2_permutation.md",
              banner="> DINOv2 paired sign-flip permutation tests on per-query Top-1, "
                     "all 40 method pairs. >=1 significant => selection signal recovered.")

    print("\n--- dinov2_similarity ---\n" + sim_df.to_string(index=False))
    print("\n--- dinov2_retrieval (boot 95% CIs) ---\n" + ci_df.to_string(index=False))
    print("\n" + "\n".join(note))
    print("\nwrote dinov2_similarity.{md,csv}, dinov2_retrieval.{md,csv}, "
          "dinov2_permutation.{md,csv}")


def _write_md(df: pd.DataFrame, path: Path, banner: str,
              note: list[str] | None = None) -> None:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    def fmt(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.4f}"
        return str(v)

    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    lines = [banner + "\n\n" + head, sep, *body]
    if note:
        lines += ["", "```", *note, "```"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
