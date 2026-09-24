#!/usr/bin/env python
"""Assemble RESULTS.md from every condition's metrics.json and per_clip.npz.

Quantitative: one table per axis, plus a paired bootstrap against condition N
over the test clips, since that is what the plan's separation rule needs and a
mean alone cannot support one.

The separation rule is applied as written in STAGE1_PLAN.md section 8, with one
honest amendment recorded in the output: with a single seed there is no
across-seed standard deviation, so the seed term cannot be evaluated and only
the paired bootstrap and the probe floor are available.

k, the DiT backbone and each condition's token shape are read from the
metrics.json files that `scripts/eval_cond.py` writes, so one script serves every
variant of the screen. `--suffix` selects a variant's tags (N1, D1, ... for
`--suffix 1`) and `--out-file` / `--comparison-dir` keep its outputs apart.

Usage:
    python scripts/make_results.py --out-root <proj>/out --proj <proj>
    python scripts/make_results.py --out-root <proj>/out --proj <proj> --suffix 1 \\
        --out-file STAGE1_K1_RESULTS.md --comparison-dir comparison_k1
"""
import argparse
import json
from pathlib import Path

import numpy as np

ORDER = ["N", "X", "D", "T", "V", "G"]
LABEL = {
    "N": "**N** null context",
    "D": "**D** Drive-JEPA domain",
    "T": "**T** Drive-JEPA task",
    "V": "**V** Traj-VAE",
    "G": "**G** V-JEPA2 ViT-H (generic)",
}
PROBE_FLOOR = 0.2739     # out/probe_vaeaug/probe_floors.json, VAE round-trip ADE
ARROW = {"higher": "↑", "lower": "↓"}

# Every metric a table reports, with the direction that counts as better and the
# format its column uses. "Best" always means best in the arrow's direction, so
# the bold cell in an error column is the smallest value, not the largest.
VISUAL_COLS = [("fvd", "lower", ".2f"), ("psnr", "higher", ".3f"),
               ("ssim", "higher", ".4f"), ("lpips", "lower", ".4f")]
TRAJ_COLS = [("ade", "lower", ".4f"), ("fde", "lower", ".4f"),
             ("dtw", "lower", ".3f"), ("heading_error_deg", "lower", ".3f"),
             ("quality_composite", "higher", ".4f")]


def _num(v):
    return isinstance(v, (int, float)) and v == v      # rejects None and NaN


def winners(runs, tags, cols):
    """Tag holding the best value of each metric, skipping ties so a column
    where every condition agrees is not given a spurious leader."""
    out = {}
    for key, direction, _ in cols:
        vals = {t: runs[t][key] for t in tags if _num(runs[t].get(key))}
        if not vals:
            continue
        pick = max if direction == "higher" else min
        best = pick(vals, key=vals.get)
        if list(vals.values()).count(vals[best]) == 1:
            out[key] = best
    return out


def cell(runs, t, key, spec, won):
    """One table cell, bold when this condition holds the column's best value."""
    v = runs[t].get(key)
    if not _num(v):
        return "-"
    s = format(v, spec)
    return f"**{s}**" if won.get(key) == t else s


def boot_delta(a, b, n=10000, seed=0):
    """Paired bootstrap of mean(a) - mean(b) over clips. Returns (delta, lo, hi, p).

    ``p`` is the two-sided fraction of resamples whose sign differs from the
    observed delta, so small p means the sign is stable under resampling."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) == 0:
        return np.nan, np.nan, np.nan, np.nan
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    obs = d.mean()
    p = float(np.mean(np.sign(boots) != np.sign(obs))) if obs != 0 else 1.0
    return obs, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--proj", required=True)
    ap.add_argument("--suffix", default="",
                    help="tag suffix of this variant's runs, e.g. 1 for N1 D1 T1 V1 G1")
    ap.add_argument("--out-file", default="RESULTS.md")
    ap.add_argument("--comparison-dir", default="comparison")
    args = ap.parse_args()
    sfx = args.suffix
    out = Path(args.out_root)
    proj = Path(args.proj)

    runs = {}
    for tag in ORDER:
        mj = out / f"eval_{tag}{sfx}" / "metrics.json"
        if mj.exists():
            runs[tag] = json.loads(mj.read_text())
            pc = out / f"eval_{tag}{sfx}" / "per_clip.npz"
            runs[tag]["_per_clip"] = dict(np.load(pc)) if pc.exists() else {}
    if not runs:
        raise SystemExit(f"no eval_*/metrics.json under {out}")
    tags = [t for t in ORDER if t in runs]
    print("conditions found:", [t + sfx for t in tags])

    ks = {int(runs[t].get("fixed_k") or 2) for t in tags}
    if len(ks) != 1:
        raise SystemExit(f"conditions disagree on k: {ks}")
    k = ks.pop()
    cutoff = 1 + (k - 1) * 4                 # MiniWan: causal, 4x temporal
    ctx_txt = "Context frame raw 0 is" if cutoff == 1 else f"Context frames raw 0 to {cutoff - 1} are"
    r0 = runs[tags[0]]
    if r0.get("dit_width"):
        dit_txt = (f"MiniDiT of width {r0['dit_width']} and {r0['dit_blocks']} blocks "
                   f"({r0['dit_params'] / 1e6:.1f}M params in {tags[0]}, the conditioned "
                   "runs adding cross-attention k/v projections sized to their encoder),")
    else:
        dit_txt = "MiniDiT"

    L = []
    A = L.append
    A("# Stage 1 results: driving representation as conditioning\n")
    A(f"One seed. Five conditions, each a {dit_txt} trained from scratch for 100")
    A(f"epochs at effective batch 32 with $k$ fixed at {k}, differing only in what feeds")
    A("the cross-attention. Evaluated on all 4,000 held-out test clips, seed 0, 50 Euler")
    A("steps. Design and rationale in `STAGE1_PLAN.md`; the trajectory extractor in")
    A("`PROBE.md`.\n")

    n_any = runs[tags[0]]["num_samples"]
    A(f"Clips evaluated: **{n_any}**. {ctx_txt} ground truth "
      "in every")
    A(f"condition; raw {cutoff} to 15 are generated.\n")
    A("Beside each metric name, ↑ means higher is better and ↓ means lower is")
    A("better, and **bold** marks the best value in that column, read in the")
    A("arrow's direction.\n")

    # ---------------- visual ----------------
    A("## 1. Visual axis\n")
    A("| condition | tokens | FVD16 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | epoch | val_flow ↓ |")
    A("|---|---|---|---|---|---|---|---|")
    won = winners(runs, tags, VISUAL_COLS)
    for t in tags:
        r = runs[t]
        tok = "-" if r.get("cond") in (None, "none") else _tok(r)
        cells = " | ".join(cell(runs, t, k, spec, won) for k, _, spec in VISUAL_COLS)
        A(f"| {LABEL.get(t, t)} | {tok} | {cells} | {r.get('epoch', '?')} | "
          f"{_vf(r)} |")
    A("")
    A("FVD16 is the primary visual metric. PSNR, SSIM and LPIPS are over the generated")
    A("frame range only; FVD16 covers the whole clip, which is the convention the")
    A("video-prediction literature uses and which `cond_eval/dit_eval.py` documents.\n")

    # ---------------- trajectory ----------------
    has_traj = any("ade" in runs[t] for t in tags)
    if has_traj:
        A("## 2. Trajectory axis\n")
        A(f"Read against the probe's VAE round-trip floor of **{PROBE_FLOOR:.4f} m**")
        A("(`PROBE.md` section 5). Differences below that floor are not readable.\n")
        A("| condition | ADE (m) ↓ | FDE (m) ↓ | DTW ↓ | heading (deg) ↓ "
          "| composite ↑ |")
        A("|---|---|---|---|---|---|")
        traj_tags = [t for t in tags if "ade" in runs[t]]
        won = winners(runs, traj_tags, TRAJ_COLS)
        for t in traj_tags:
            cells = " | ".join(cell(runs, t, k, spec, won) for k, _, spec in TRAJ_COLS)
            A(f"| {LABEL.get(t, t)} | {cells} |")
        A("")
        A("ADE is the primary trajectory metric. DTW absorbs a shift along the direction")
        A("of travel, so a condition with the right path at the wrong speed shows in ADE")
        A("and not in DTW. The composite is a guard against degenerate motion rather than")
        A("a discriminator, being near its ceiling on real motion.\n")

    # ---------------- paired bootstrap vs N ----------------
    if "N" in runs:
        A("## 3. Paired bootstrap against N\n")
        A("10,000 resamples over the shared test clips. A negative delta favours the")
        A("condition for an error metric. `p` is the fraction of resamples whose sign")
        A("differs from the observed delta.\n")
        for metric, better in (("fvd", None), ("psnr", "higher"), ("lpips", "lower"),
                               ("ade", "lower"), ("dtw", "lower")):
            key = {"fvd": None}.get(metric, metric)
            if key is None:
                continue          # FVD is a distribution distance: no per-clip value
            rows = []
            for t in tags:
                if t == "N":
                    continue
                a = runs[t]["_per_clip"].get(key)
                b = runs["N"]["_per_clip"].get(key)
                if a is None or b is None or len(a) != len(b):
                    continue
                d, lo, hi, p = boot_delta(a, b)
                sig = "yes" if (lo > 0) == (hi > 0) else "no"
                rows.append(f"| {LABEL.get(t, t)} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] "
                            f"| {p:.4f} | {sig} |")
            if rows:
                A(f"**{metric.upper()} {ARROW[better]}**, {better} is better\n")
                A("| condition | delta vs N | 95% CI | p | CI excludes 0 |")
                A("|---|---|---|---|---|")
                L.extend(rows)
                A("")
        A("FVD16 has no per-clip value, being a distance between distributions, so it")
        A("cannot be bootstrapped this way and is compared on its point estimate alone.\n")

    # ---------------- qualitative ----------------
    comp = out / args.comparison_dir
    sheets = sorted(comp.glob("sheet_clip*.png")) if comp.exists() else []
    gifs = sorted(comp.glob("clip*_GT_vs_*.gif")) if comp.exists() else []
    A("## 4. Qualitative comparison\n")
    if sheets:
        A("Ground truth and every condition on the same clips, same seed, same context")
        A(f"frames. Row order is in `out/{args.comparison_dir}/ROW_ORDER.txt`.\n")
        for s in sheets[:3]:
            A(f"![{s.stem}](out/{args.comparison_dir}/{s.name})\n")
        A(f"All {len(sheets)} sheets and {len(gifs)} animated versions are in")
        A(f"`out/{args.comparison_dir}/`. The GIFs are the more useful of the two for judging")
        A("temporal coherence.\n")
    else:
        A("Not generated yet; run `scripts/node_comparison.sh`.\n")

    # ---------------- caveats ----------------
    A("## 5. How to read this\n")
    A("**One seed, so the plan's separation rule is only partly applicable.**")
    A("`STAGE1_PLAN.md` section 8 requires a gap exceeding twice the pooled across-seed")
    A("standard deviation, which cannot be evaluated without repeats. What is available")
    A("is the paired bootstrap over clips and the probe floor, and both are reported")
    A("above. A bootstrap CI excluding zero says the ordering is stable across clips,")
    A("not that it is stable across training runs.\n")
    shape = {t: _shape(runs[t]) for t in tags if runs[t].get("cond") not in (None, "none")}
    if all(shape.get(t) for t in ("V", "D", "T", "G")):
        nums = {t: shape[t][0] * shape[t][1] for t in shape}
        lo, hi = nums["D"] / nums["V"], nums["G"] / nums["V"]
        A("**Conditioning bandwidth is not controlled.** The conditions differ enormously in")
        A(f"how much they feed the cross-attention: V is {shape['V'][0]} tokens of width "
          f"{shape['V'][1]}, so {nums['V']:,}")
        A(f"numbers, while D and T are {shape['D'][0]} x {shape['D'][1]} and G is "
          f"{shape['G'][0]} x {shape['G'][1]}, so {nums['D']:,} and {nums['G']:,}.")
        A(f"That is a factor of {lo:.0f} to {hi:.0f}. If the high-bandwidth conditions "
          "lead, capacity is")
        A("as good an explanation as driving-specificity, and the D-against-T comparison is")
        A("the one that isolates domain from task at matched bandwidth.\n")
    A(f"**$k$ is fixed at {k}**, which departs from the sampled $k$ that produced")
    A("`dit10m32b100e`. That is why N was retrained here rather than reused; all five")
    A("share the fixed-$k$ distribution and so are comparable to each other, but none is")
    A("comparable to that earlier checkpoint.")
    if k == 1:
        A("With one context frame the encoders see raw frame 0 alone (twice over for the")
        A("ViTs, whose tubelet needs an even count), so no condition carries any motion")
        A("from the context: every movement in the generated frames is inferred from a")
        A("single still.")
    A("")

    (proj / args.out_file).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {proj / args.out_file} ({len(L)} lines)")


def _shape(r):
    """(tokens, width) of a condition's cross-attention input, from its metrics.
    Evals that predate those fields are the k=2 runs, read off the run name."""
    if r.get("cond_tokens"):
        return int(r["cond_tokens"]), int(r["cond_dim"])
    pc = str(r.get("cond_run", ""))
    if "vith" in pc:
        return 768, 1280
    if "vitl" in pc:
        return 768, 1024
    if r.get("cond") == "traj_vae":
        return 432, 16
    return None


def _tok(r):
    sh = _shape(r)
    if sh:
        return f"{sh[0]} x {sh[1]}"
    pc = r.get("cond_run", "")
    if "vith" in str(pc):
        return "768 x 1280"
    if "vitl" in str(pc):
        return "768 x 1024"
    if r.get("cond") == "traj_vae":
        return "432 x 16"
    return "?"


def _vf(r):
    v = r.get("val_flow")
    return f"{v:.6f}" if isinstance(v, (int, float)) else "-"


if __name__ == "__main__":
    main()
