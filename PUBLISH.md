# Publishing this repository

The release is committed and ready. What remains is creating the remote and pushing, which needs
an account this environment does not have access to.

Current state: one commit on `main`, 57 files, 0.58 MB, clean working tree.

## Option A — GitHub

1. Create an empty repository at <https://github.com/new>. Name it e.g. `hmtinet`. Do **not** add a
   README, licence or .gitignore (this repository already has them).
2. From inside this directory:

```
git remote add origin https://github.com/<your-account>/hmtinet.git
git push -u origin main
```

The resulting URL is `https://github.com/<your-account>/hmtinet`.

## Option B — Gitee (faster inside mainland China)

1. Create an empty repository at <https://gitee.com/projects/new>.
2. Same two commands with the Gitee URL.

## Then, in the manuscript

Replace the placeholder in the Data Availability Statement of `original_tex.tex`:

```
\url{[TODO: repository URL]}
```

with the real URL, rebuild both PDFs, and repackage. Nothing else in the manuscript refers to the
repository.

## Optional: make the archive citable

Zenodo can mint a DOI for a GitHub release, which is a more durable identifier than a repository
URL and is what MDPI prefers where available:

1. Sign in to <https://zenodo.org> with GitHub.
2. Enable the repository in Zenodo's GitHub settings.
3. Publish a GitHub release; Zenodo archives it and issues a DOI.

If a DOI is obtained, cite that in the Data Availability Statement instead of the bare URL.

## Commit identity

The commit is authored as `Xuan Zhang <zhangxuan@gzhs.edu.cn>`, set repository-locally so that the
public history does not carry this machine's account. To change it before publishing:

```
git config user.name "…"
git config user.email "…"
git commit --amend --reset-author --no-edit
```

## Path handling

Scripts resolve paths through `paths.py` at the repository root; none contain absolute paths. Harvested inputs are located via the `HMTINET_DATA` environment variable, since they are not redistributed. See README.md, section 'Re-running the pipeline'.

## What is in the repository, and what is not

Included: pipeline code, derived artefacts, the manuscript's tables as CSV, and a verification
script that recomputes the headline numbers from the artefacts alone.

Not included: subtitle tracks, which are third-party material. `data/track_manifest.csv` gives the
path, extension, byte size and SHA-256 of each of the 19 tracks actually used, so the same files
can be verified once re-obtained.

The annotation sheets do contain short dialogue excerpts (968 cues of 24,557). They are needed to
check the reported agreement statistic and are labelled as research material in the README.

