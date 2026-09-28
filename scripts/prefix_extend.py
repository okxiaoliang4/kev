"""Incremental state prefix on MLX: a state that grows at its end, served with KEV_PREFIX_EXTEND on and off, on this Mac.

    uv run --extra serve python scripts/prefix_extend.py --run jaredpalmer/kev-0.8b --out runs/prefix-extend-0.8b/report.json
    uv run --extra serve python scripts/prefix_extend.py --run jaredpalmer/kev-4b --out runs/prefix-extend-4b/report.json

The workload is a log that grows: step k's state is the first k clean development records of decision-v7 joined by blank
lines, and step k asks the questions of record k (the item just appended), the shape of an agent transcript or a ticket
thread scored after every turn. Both arms run what `kev.serve` runs for one request (PrefixCache.plan, bases when the
arm extends, model.probs_batch, PrefixCache.store) over the same steps, from an empty cache; time is model time.

Decision rule, fixed before the first run: extension passes if every question's probabilities are within 0.04 of the
from-scratch arm's (the bar tests/test_mlx.py sets for the prefix form against the row form: same function, different bf16
computations) and an argmax differs only where the from-scratch top-2 margin is under 0.04. Also, every 10th step, the
cached prefix a step extends must still answer its own record bit for bit after the extension (same kernels, same
shapes: tests/test_mlx.py's rule for a reused prefix), so extending never writes into the cached entry.
"""
import argparse, json, statistics, time
from pathlib import Path

import mlx.core as mx
import torch

from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE
from kev.serve import PrefixCache
from kev.suite import load_split, write_json

BAR = TIE = 0.04


def run_arm(model, encs, extend):
    """-> (per-step model ms, per-step probabilities, per-step state tokens reused, untouched-prefix checks, peak MB)."""
    cache = PrefixCache(4, model.prefix_min_tokens)
    ms, probs, reused, untouched = [], [], [], []
    mx.reset_peak_memory()
    for k, enc in enumerate(encs):
        check = extend and k > 0 and k % 10 == 0
        if check:                                    # the previous step's cached prefix, read before this step extends it
            prev = cache.entries.get(cache.plan([encs[k - 1]])[0][0])
            before = model.probs_with_prefix(encs[k - 1], prev) if prev is not None else None
        t = time.perf_counter()
        keys, cached, keep = cache.plan([enc])
        bases = cache.bases(model, keys, cached) if extend else None
        ps, prefixes = model.probs_batch([enc], cached, keep, bases) if extend else model.probs_batch([enc], cached, keep)
        cache.store(keys, cached, prefixes)
        ms.append((time.perf_counter() - t) * 1000)
        probs.append(ps[0]); reused.append(bases[0][0] if bases and bases[0] is not None else 0)
        if check and before is not None:
            after = model.probs_with_prefix(encs[k - 1], prev)
            untouched.append({"step": k, "extended": bool(reused[-1]), "bit_identical": all(torch.equal(x, y) for x, y in zip(before, after))})
    return ms, probs, reused, untouched, round(mx.get_peak_memory() / 2**20)


def quartiles(xs):
    q = len(xs) // 4
    return [round(statistics.median(xs[i * q:(i + 1) * q if i < 3 else len(xs)]), 1) for i in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b"); ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--suite", default="evals/v7/decision-v7"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    recs = [materialize(r) for r in load_split(a.suite, "development") if r["_meta"]["variant"] == "clean"][: a.steps]
    tok, model = Checkpoint(a.run).load("mps", LoadOptions(backend="mlx"))
    if not getattr(model, "extends_prefix", False): raise SystemExit(f"{a.run} does not load on a backend that extends prefixes")
    encs = [model.encode(tok, {"state": "\n\n".join(r["state"] for r in recs[:k + 1]), "questions": recs[k]["questions"]},
                         max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH) for k in range(len(recs))]
    run_arm(model, encs[:3], True)                                   # warm the kernels for the first shapes
    on_ms, on_p, reused, untouched, on_peak = run_arm(model, encs, True)
    off_ms, off_p, _, _, off_peak = run_arm(model, encs, False)

    dps, flips, bad = [], 0, []
    for k, (ps, rs) in enumerate(zip(on_p, off_p)):
        for p, r in zip(ps, rs):
            d = float((p - r).abs().max()); dps.append(d)
            if p.argmax() != r.argmax():
                flips += 1
                top = r.topk(2).values
                if float(top[0] - top[1]) >= TIE: bad.append({"step": k, "reason": "argmax flip on a decided question", "margin": float(top[0] - top[1])})
            if d >= BAR: bad.append({"step": k, "reason": "max |dp| over the bar", "dp": d})
    states = [enc["seg"].count(0) for enc in encs]
    report = {
        "run": a.run, "suite": f"{a.suite} development (clean)", "steps": len(encs), "questions": len(dps),
        "state_tokens": {"first": states[0], "last": states[-1], "median_added_per_step": statistics.median(b - c for b, c in zip(states[1:], states))},
        "extended_steps": sum(1 for r in reused if r), "reused_state_tokens": sum(reused),
        "parity_vs_scratch": {"bar": BAR, "tie": TIE, "max_dp": max(dps), "mean_dp": statistics.mean(dps), "argmax_flips": flips, "violations": bad, "pass": not bad},
        "cached_prefix_untouched": {"checks": len(untouched), "extended": sum(u["extended"] for u in untouched),
                                    "bit_identical": sum(u["bit_identical"] for u in untouched), "pass": bool(untouched) and all(u["bit_identical"] for u in untouched)},
        "peak_memory_mb": {"extend_on": on_peak, "extend_off": off_peak},
        "model_ms": {"extend_on": {"median": round(statistics.median(on_ms), 1), "by_quarter": quartiles(on_ms), "total_s": round(sum(on_ms) / 1000, 1)},
                     "extend_off": {"median": round(statistics.median(off_ms), 1), "by_quarter": quartiles(off_ms), "total_s": round(sum(off_ms) / 1000, 1)}},
        "dtype": model.dtype,
    }
    print(json.dumps(report, indent=1))
    if a.out: Path(a.out).parent.mkdir(parents=True, exist_ok=True); write_json(Path(a.out), report)


if __name__ == "__main__":
    main()
