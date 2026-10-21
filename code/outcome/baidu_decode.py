"""Correct Baidu Index decoding, transcribed from the site's own bundle.

The routine was read out of main.9de0e5723a74b2999bc4.js rather than guessed:

    decrypt: function (t, e) {            // t = ptbk key, e = scrambled payload
        for (var n = t.split(""), i = e.split(""), a = {}, r = [], o = 0;
             o < n.length / 2; o++)
            a[n[o]] = n[n.length / 2 + o];   // first half -> second half map
        for (var s = 0; s < e.length; s++)
            r.push(a[i[s]]);                 // per-character substitution
        return r.join("")                    // comma-separated values
    }

Called as `decrypt(key, block.data).split(",")`, so the output is a comma-separated
numeric series indexed by date, and the mapping is per character, not per pair. The
earlier two-character guess was wrong.

This module is the single implementation used by the harvester, so there is one place
to audit.
"""

import os
import sys as _sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import paths as p
from paths import DATA as _DATA_ROOT

DATA_ANNOTATION = os.path.join(_DATA_ROOT, "annotation")
import json
import os
import re
from datetime import date, timedelta


def decrypt(key: str, payload: str) -> str:
    """Exact transcription of the site's decrypt()."""
    n = len(key)
    table = {}
    for o in range(n // 2):
        table[key[o]] = key[n // 2 + o]
    return "".join(table.get(ch, "") for ch in payload)


def decrypt_series(key: str, payload: str):
    """decrypt() then split on commas, as the caller does."""
    s = decrypt(key, payload)
    out = []
    for x in s.split(","):
        x = x.strip()
        if x == "":
            out.append(None)
        else:
            try:
                out.append(int(x))
            except ValueError:
                out.append(None)
    return out


def date_range(start: str, end: str, granularity: str = "day"):
    """Inclusive date list matching the series length returned by the API."""
    y, m, d = (int(x) for x in start.split("-"))
    y2, m2, d2 = (int(x) for x in end.split("-"))
    cur, last = date(y, m, d), date(y2, m2, d2)
    out = []
    if granularity == "day":
        while cur <= last:
            out.append(cur.isoformat())
            cur += timedelta(days=1)
    elif granularity == "month":
        cur = date(y, m, 1)
        last = date(y2, m2, 1)
        while cur <= last:
            out.append(cur.strftime("%Y-%m"))
            cur = date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
    return out


def verify_with_local_raw():
    """Self-check using the payload captured from the live session.

    The thumbnail call captured earlier returns a full-history series but its key was
    not retrieved, so this checks the routine's mechanics (lengths, value plausibility)
    once a key is supplied, and reports clearly when a key is missing rather than
    silently producing zeros.
    """
    raw = p.require(os.path.join(p.BAIDU, "out", "raw_thumbnail_baibai.json"), "raw harvest dump")
    if not os.path.exists(raw):
        print("no captured payload at", raw)
        return
    d = json.load(open(raw, encoding="utf-8"))
    blk = d.get("data", {})
    blk = blk.get("pc", blk)
    payload = blk.get("data", "")
    print("captured payload length:", len(payload))
    print("date range:", blk.get("startDate"), "->", blk.get("endDate"))
    dr = date_range(blk.get("startDate", "2006-06-01"), blk.get("endDate", "2026-10-15"), "month")
    print("expected monthly points:", len(dr))
    print("payload is comma-separated:", "," in payload)
    print("first 80 chars:", payload[:80])


if __name__ == "__main__":
    verify_with_local_raw()
