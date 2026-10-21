"""Validate all series with a coverage-based acceptance rule.

Replaces the earlier all-or-nothing test. A window is accepted when most of its days are
present, reporting the exact count, because two of the harvests start or end a few days
short of the ideal span (厨子戏子痞子 lacks the first week of the baseline; 一九四二 lacks the
last three weeks of the outcome window). Discarding them for a few missing days would throw
away usable evidence; sweeping them in silently would overstate coverage. Both are wrong,
so the counts are reported per window and the ratio is computed on the days that exist.

八佰 is read from resweep.json, which stores a bare date->value mapping rather than the
{"points": [...]} shape the other harvests use.
"""

import os
import sys as _sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import paths as p
from paths import DATA as _DATA_ROOT

DATA_ANNOTATION = os.path.join(_DATA_ROOT, "annotation")
import datetime as dt
import glob
import json
import math
import os
from collections import Counter

DIRS = [os.path.join(p.BAIDU, "resweep"), os.path.join(p.BAIDU, "final"),
        os.path.join(p.BAIDU, "ready"), os.path.join(p.BAIDU, "pre2016"),
        os.path.join(p.BAIDU, "tabs3"), os.path.join(p.BAIDU, "tabs2")]
OUT = p.ensure_dir(os.path.join(p.DATA, "outcome"))
os.makedirs(OUT, exist_ok=True)
MIN_WINDOW_FRACTION = 0.55   # below this the ratio is not trustworthy; above it, report as partial

RELEASES = {
    "八佰": "2020-08-21", "金陵十三钗": "2011-12-15", "铁道英雄": "2021-11-19",
    "铁道飞虎": "2016-12-23", "厨子戏子痞子": "2013-03-29", "一九四二": "2012-11-29",
    "战争子午线": "1990-01-01",
}
EN = {"八佰": "The Eight Hundred", "金陵十三钗": "The Flowers of War",
      "铁道英雄": "Railway Heroes", "铁道飞虎": "Railroad Tigers",
      "厨子戏子痞子": "The Chef, the Actor, the Scoundrel", "一九四二": "Back to 1942"}


def series_from(path):
    """Accept {"points":[...]}, {"series":{...}} or a bare {date: value} mapping."""
    try:
        j = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(j, dict):
        return None
    pts = j.get("points")
    if pts:
        s = {x["date"]: x["value"] for x in pts if x.get("date")}
        return s or None
    s = j.get("series")
    if isinstance(s, dict) and s:
        return {k: v for k, v in s.items() if isinstance(v, (int, float))}
    if j and all(isinstance(k, str) and len(k) == 10 and k[4] == "-" for k in j):
        return {k: v for k, v in j.items() if isinstance(v, (int, float))}
    return None


def candidates(film):
    out = []
    for d in DIRS:
        for fp in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            if film not in os.path.basename(fp):
                continue
            s = series_from(fp)
            if s:
                out.append((fp, s))
    # 八佰 lives in resweep.json with no film name in the filename
    if film == "八佰":
        resweep_path = os.path.join(p.BAIDU, "resweep", "resweep.json")
        if os.path.isfile(resweep_path):
            s = series_from(resweep_path)
            if s:
                out.append((resweep_path, s))
    return out


def window(series, rel, d0, d1):
    vals, cur, end = [], rel + dt.timedelta(days=d0), rel + dt.timedelta(days=d1)
    while cur <= end:
        s = cur.isoformat()
        if s in series:
            vals.append(series[s])
        cur += dt.timedelta(days=1)
    return vals


def main():
    usable, excluded = [], []
    for film, rel_iso in sorted(RELEASES.items(), key=lambda kv: kv[1]):
        rel = dt.date.fromisoformat(rel_iso)
        best = None
        for fp, s in candidates(film):
            base = window(s, rel, -84, -28)
            outc = window(s, rel, 0, 56)
            fb, fo = len(base) / 57, len(outc) / 57
            score = (min(fb, fo), 1 if any(v > 0 for v in s.values()) else 0, len(s))
            if best is None or score > best[0]:
                best = (score, fp, s, base, outc)
        if best is None:
            excluded.append((film, "no series found"))
            continue
        score, src, series, base, outc = best
        if not score[1]:
            excluded.append((film, "all values zero: no search volume for this release"))
            continue
        if score[0] < MIN_WINDOW_FRACTION:
            excluded.append((film, f"window coverage {score[0]:.0%} below {MIN_WINDOW_FRACTION:.0%}"))
            continue

        ks = sorted(series)
        gaps = Counter()
        for a, b in zip(ks, ks[1:]):
            gaps[(dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days] += 1
        mg = max(gaps, key=gaps.get)
        gran = "daily" if mg <= 1 else "near-daily" if mg <= 3 else "weekly" if mg <= 9 else "sparse"

        mb, mo = sum(base) / len(base), sum(outc) / len(outc)
        lr = math.log(mo) - math.log(mb) if mb > 0 and mo > 0 else None
        pk = max(series, key=lambda k: series[k])
        pd = (dt.date.fromisoformat(pk) - rel).days

        rec = {"film": film, "english": EN.get(film, film), "release": rel_iso,
               "source_file": os.path.relpath(src, p.EXTERNAL),
               "granularity": gran, "mode_gap_days": mg, "points": len(series),
               "first": ks[0], "last": ks[-1],
               "peak": {"date": pk, "value": series[pk], "days_from_release": pd},
               "baseline": {"days": [-84, -28], "n": len(base), "expected": 57,
                            "coverage": round(len(base) / 57, 3), "mean": round(mb, 2)},
               "outcome": {"days": [0, 56], "n": len(outc), "expected": 57,
                           "coverage": round(len(outc) / 57, 3), "mean": round(mo, 2)},
               "log_ratio": round(lr, 4) if lr is not None else None,
               "ratio": round(mo / mb, 1) if mb > 0 else None,
               "series": series,
               "caveat": "Query term is the film title; the manuscript's proxy is heritage-site "
                         "attention. Same pipeline, different estimand."}
        json.dump(rec, open(os.path.join(OUT, f"{film}_proxy.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        usable.append(rec)
        print(f"{film:<14} {rel_iso} {gran:<6} peak{pd:+3d}d  base {mb:>10,.0f} "
              f"({len(base):>2}/57)  out {mo:>10,.0f} ({len(outc):>2}/57)  {mo/mb:>7.1f}x")

    print("\nexcluded:")
    for f, why in excluded:
        print(f"  {f:<14} {why}")
    json.dump({"usable": usable, "excluded": [{"film": f, "reason": w} for f, w in excluded]},
              open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nusable: {len(usable)} of {len(RELEASES)} | wrote {OUT}")


if __name__ == "__main__":
    main()

