"""Reproduce the ranking-stability statistics quoted in the manuscript, and persist them.

The manuscript states, for 2000 bootstraps of the nineteen films under its own executed intensity
definition (gated, recurrence >= 2, length-normalised cue share):
  - the full-sample top-five set is recovered in 31.2% of resamples
  - mean Kendall tau against the full-sample order is 0.834, minimum 0.238

Recomputing gives a markedly lower exact-recovery rate, so this script computes all three
quantities with a large bootstrap, writes them to an artefact, and reports the comparison so the
manuscript can be corrected to a reproducible value.
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
import random
from collections import defaultdict

TRAIN = p.require(p.WORK, "working directory with cues.json")
OUT = os.path.join(p.ensure_dir(p.RESULTS_STABILITY), "ranking_stability_bootstrap.json")

cues = json.load(open(os.path.join(TRAIN, "cues.json"), encoding="utf-8"))
man = json.load(open(os.path.join(TRAIN, "manifest.json"), encoding="utf-8"))
labels = man["labels"]

NO_WIKIDATA = {"Landmine warfare", "Guerrilla warfare", "National humiliation",
               "Taierzhuang Battle Memorial", "Pingxingguan Memorial", "Red star insignia",
               "Rising sun flag", "Nationalist flag"}

films = sorted({c["film"] for c in cues})
n_films = len(films)
cpf = defaultdict(int)
pos = defaultdict(int)
for c in cues:
    cpf[c["film"]] += 1
    for j, v in enumerate(c["labels"]):
        if v:
            pos[(c["film"], j)] += 1

KEYS = [j for j in range(len(labels)) if labels[j] not in NO_WIKIDATA]
I = {}
for f in films:
    for j in KEYS:
        c = pos.get((f, j), 0)
        I[(f, j)] = (c / cpf[f]) if c >= 2 else 0.0


def mean_of(sample):
    agg = {j: 0.0 for j in KEYS}
    for f in sample:
        for j in KEYS:
            agg[j] += I[(f, j)]
    return {j: v / len(sample) for j, v in agg.items()}


def order(m):
    return [j for j, _ in sorted(m.items(), key=lambda kv: (-kv[1], kv[0]))]


def kendall_tau(a, b):
    """Kendall tau-b between two full orders."""
    n = len(a)
    posa = {v: i for i, v in enumerate(a)}
    posb = {v: i for i, v in enumerate(b)}
    conc = disc = 0
    for x in range(n):
        for y in range(x + 1, n):
            p, q = a[x], a[y]
            s = (posa[p] - posa[q]) * (posb[p] - posb[q])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    d = conc + disc
    return (conc - disc) / d if d else 1.0


obs_means = mean_of(films)
obs_order = order(obs_means)
obs_top = frozenset(obs_order[:5])

N = 4000
rng = random.Random(20261020)
hits = 0
taus = []
for _ in range(N):
    sample = [rng.choice(films) for _ in range(n_films)]
    m = mean_of(sample)
    o = order(m)
    if frozenset(o[:5]) == obs_top:
        hits += 1
    taus.append(kendall_tau(obs_order, o))

rate = hits / N
mean_tau = sum(taus) / len(taus)
min_tau = min(taus)

print("observed top-5:", [labels[j] for j in obs_top])
print()
print("%-42s %10s %10s" % ("statistic", "recomputed", "manuscript"))
print("%-42s %10.4f %10.4f" % ("exact top-5 recovery rate", rate, 0.312))
print("%-42s %10.4f %10.4f" % ("mean Kendall tau vs full-sample order", mean_tau, 0.834))
print("%-42s %10.4f %10.4f" % ("minimum Kendall tau over resamples", min_tau, 0.238))

res = {
    "method": ("film-clustered nonparametric bootstrap over the 19 films; intensity is the "
               "length-normalised cue share under the executed rule (gate = dedicated Wikidata "
               "entity, recurrence >= 2 cues in the film); the top-5 set is compared for exact "
               "equality; Kendall tau-b is computed against the full-sample ordering of the "
               "gated concept pool"),
    "n_films": n_films,
    "n_bootstrap": N,
    "seed": 20261020,
    "concepts_considered": [labels[j] for j in sorted(KEYS)],
    "observed_top5": [labels[j] for j in obs_top],
    "exact_top5_recovery_rate": rate,
    "mean_kendall_tau": mean_tau,
    "min_kendall_tau": min_tau,
    "manuscript_values": {"exact_top5_recovery_rate": 0.312,
                          "mean_kendall_tau": 0.834,
                          "min_kendall_tau": 0.238},
    "discrepancy_note": ("The exact top-5 recovery rate recomputes below the value printed in the "
                         "manuscript. The qualitative conclusion is unchanged and in fact "
                         "strengthened: the ranking is not stable enough to support element-level "
                         "ordering claims."),
}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\nwrote", OUT)
