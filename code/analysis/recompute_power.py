"""Recompute the power-analysis requirements for the corrected specification.

The ontology actually implemented has FOUR categories, so the grouped confirmatory
specification has four category-level regressors plus four controls (p = 8), not
six plus four (p = 10). The required sample size is

    F = ceil( (z_{1-a/2} + z_{1-b})^2 * (1 - r^2) / r^2 ) + p + 1

with power 0.8, alpha = 0.05, and the multiplicity correction shown.
"""
import math

def z(p):
    # inverse normal CDF via Acklam's rational approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def required(r, alpha, p):
    zc = z(1 - alpha / 2) + z(0.8)
    return math.ceil(zc ** 2 * (1 - r ** 2) / r ** 2) + p + 1


SPECS = [
    ("four-category model, alpha=0.05", 0.05, 8),
    ("four-category model, Bonferroni/4", 0.05 / 4, 8),
    ("leaf model (24 concepts), alpha=0.05", 0.05, 28),
    ("leaf model (24 concepts), Bonferroni/24", 0.05 / 24, 28),
    ("leaf model (24 concepts), Bonferroni/24, as previously computed with p=8", 0.05 / 24, 8),
]

print("z(1-0.025)=%.4f  z(0.8)=%.4f" % (z(0.975), z(0.8)))
print()
for r, name in [(0.3, "r=0.3 (medium)"), (0.4, "r=0.4"), (0.5, "r=0.5 (large)")]:
    print(name)
    for label, alpha, p in SPECS:
        print("   %-62s p=%-3d F=%d" % (label, p, required(r, alpha, p)))
    print()

print("continuous curves for the figure (p=8 and p=28 for the leaf model):")
for r in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    print("  r=%.2f  none(p8)=%3d  cat4(p8)=%3d  leaf24(p28)=%3d" % (
        r, required(r, 0.05, 8), required(r, 0.05 / 4, 8), required(r, 0.05 / 24, 28)))
