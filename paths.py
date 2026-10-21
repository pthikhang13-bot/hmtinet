"""Path resolution for the HMTI-Net release.

Every script resolves its inputs and outputs through this module instead of an absolute path, so
the release runs wherever it is cloned.

Two kinds of input are distinguished:

  * Artefacts shipped with the release live under the repository root: ``data/``, ``results/`` and
    ``tables/``. These are always present after a clone.

  * Harvested material is not redistributed: subtitle tracks, the raw search-index dumps, and the
    intermediate files built from them (``cues.json``, annotation working sheets, model checkpoints).
    Those are looked up under ``HMTINET_DATA``, which defaults to ``<repo>/_external``. Point it at
    a directory you populated yourself:

        HMTINET_DATA=/path/to/hmtinet-data python code/extract/extract.py

      with the layout

        $HMTINET_DATA/
          subtitles/    one directory per film, holding the .ass/.srt track
          baidu/        raw harvest dumps by batch
          work/         intermediates written by the pipeline

    ``data/track_manifest.csv`` gives the size and SHA-256 of each track that was used, so a
    re-obtained file can be checked before it is used.

Anything that needs a missing input raises SystemExit with the path and what to do about it, rather
than failing with a bare FileNotFoundError deep inside a loop.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Shipped with the release.
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(ROOT, "tables")
CODE = os.path.join(ROOT, "code")
SCRIPTS = os.path.join(ROOT, "scripts")

# Structured outputs, grouped as they appear under results/.
RESULTS_EXTRACTION = os.path.join(RESULTS, "extraction")
RESULTS_SCREENING = os.path.join(RESULTS, "screening")
RESULTS_AGREEMENT = os.path.join(RESULTS, "agreement")
RESULTS_DETECTORS = os.path.join(RESULTS, "detectors")
RESULTS_STABILITY = os.path.join(RESULTS, "stability")
RESULTS_REFERENCES = os.path.join(RESULTS, "references")

# Not redistributed; supplied by the user.
EXTERNAL = os.environ.get("HMTINET_DATA") or os.path.join(ROOT, "_external")
SUBTITLES = os.path.join(EXTERNAL, "subtitles")
BAIDU = os.path.join(EXTERNAL, "baidu")
WORK = os.path.join(EXTERNAL, "work")

_HINT = (
    "\n\nThis input is not redistributed with the release because it is third-party material.\n"
    "Set HMTINET_DATA to a directory containing subtitles/, baidu/ and work/, then re-run.\n"
    "See README.md, section 'Re-running the pipeline'."
)


def require(path, what):
    """Return ``path`` if it exists, otherwise stop with an actionable message."""
    if not os.path.exists(path):
        raise SystemExit("%s not found:\n  %s%s" % (what, path, _HINT))
    return path


def require_external():
    """Stop early if the external input tree is absent, before any per-file lookup."""
    if not os.path.isdir(EXTERNAL):
        raise SystemExit(
            "External input directory not found:\n  %s\n"
            "The pipeline needs the harvested subtitle tracks, which are not redistributed.\n"
            "Set HMTINET_DATA to a populated directory and re-run.%s" % (EXTERNAL, _HINT))
    return EXTERNAL


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
