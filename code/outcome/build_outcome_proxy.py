"""Build the consumption proxy for 八佰 from the harvested Baidu Index series.

The manuscript defines the consumption proxy as heritage-site search attention over a
pre-release baseline window and a post-release outcome window. Baidu Index cannot index
heritage sites under study here without a keyword mapping that has not been fixed, so
this file builds the *film-term* series instead and labels it as what it is: film-level
search attention. That distinction is carried into the paper rather than glossed over.

Windows follow the manuscript:
    baseline  : 12 to 4 weeks before release
    outcome   : 0 to 8 weeks after release
    estimand  : log(outcome mean) - log(baseline mean)

Coverage is reported, because the harvest does not span every day in the window.
"""

import os
import sys as _sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import paths
from paths import DATA as _DATA_ROOT

DATA_ANNOTATION = os.path.join(_DATA_ROOT, "annotation")
import datetime as dt
import json
import math
import os

SRC = paths.require(os.path.join(paths.BAIDU, "resweep", "resweep.json"), "resweep harvest")
FILM = "八佰"
RELEASE = "2020-08-21"
OUT = paths.ensure_dir(os.path.join(paths.DATA, "outcome"))
os.makedirs(OUT, exist_ok=True)


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    pts = {p["date"]: p["value"] for p in d["points"]}
    rel = dt.date.fromisoformat(RELEASE)

    def window(d0, d1):
        out = []
        cur = rel + dt.timedelta(days=d0)
        end = rel + dt.timedelta(days=d1)
        while cur <= end:
            s = cur.isoformat()
            if s in pts:
                out.append((s, pts[s]))
            cur += dt.timedelta(days=1)
        return out

    expect_base = 8 * 7          # 12w-4w = 56 days
    expect_out = 8 * 7 + 1       # 0-8w inclusive
    base = window(-84, -28)
    out = window(0, 56)

    mb = sum(v for _, v in base) / len(base) if base else float("nan")
    mo = sum(v for _, v in out) / len(out) if out else float("nan")
    estimand = math.log(mo) - math.log(mb) if base and out else None

    rec = {
        "film": FILM,
        "release": RELEASE,
        "source": "Baidu Index, search-term series, harvested from the rendered chart",
        "term": FILM,
        "proxy_definition": "film-level search attention; NOT heritage-site attention",
        "windows": {
            "baseline_days": [-84, -28],
            "outcome_days": [0, 56],
        },
        "coverage": {
            "baseline_points": len(base), "baseline_expected": expect_base,
            "outcome_points": len(out), "outcome_expected": expect_out,
        },
        "baseline_mean": round(mb, 2),
        "outcome_mean": round(mo, 2),
        "log_ratio": round(estimand, 4) if estimand is not None else None,
        "series": dict(sorted(pts.items())),
        "windows_detail": {"baseline": base, "outcome": out},
        "caveats": [
            "The harvested series covers about 91% of days in the requested range; the "
            "window means are computed over observed days only.",
            "Baidu Index provides daily resolution only from 2016 onward, which restricts "
            "the impact model to three films of the assembled corpus.",
            "The property targeted by the manuscript's proxy is heritage-site attention; "
            "this file measures film-term attention, so it validates the pipeline and the "
            "magnitude of a release effect rather than the study's exact estimand.",
            "Single film: this is an illustration of the pipeline, not an estimate with a "
            "sampling distribution.",
        ],
    }
    json.dump(rec, open(os.path.join(OUT, f"{FILM}_consumption_proxy.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"film {FILM}, release {RELEASE}")
    print(f"baseline (weeks -12..-4): {len(base)}/{expect_base} days, mean {mb:,.0f}")
    print(f"outcome  (weeks 0..+8):   {len(out)}/{expect_out} days, mean {mo:,.0f}")
    print(f"log ratio (outcome/baseline): {estimand:.3f}  => ratio {mo/mb:.1f}x"
          if estimand is not None else "insufficient coverage")
    print(f"\nwrote {os.path.join(OUT, FILM + '_consumption_proxy.json')}")


if __name__ == "__main__":
    main()
