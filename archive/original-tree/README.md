# Original working tree, verbatim

Every source file from the top level of the AirSim working directory on the
machine where the Chapter 7 work was finished (September 2026): 226 Python
scripts, 5 shell drivers, 14 AirSim `settings*.json` variants, the Portuguese
notes and results documents, and the auxiliary JSON/YAML/TXT files — 260 files,
copied as they were, with no edits.

It is here for completeness and provenance, not to be run. The rest of this
repository is the curated version of the same code: most of these scripts appear
there renamed, moved into packages, translated and with portable paths
(`detector_frustum_pn2.py` → `detection/frustum/detector.py`,
`ekf_3d_global.py` → `tracking/ekf_3d.py`, …), and one representative of each
family of early experiments is in `archive/` next to this folder. Everything else
— the connection tests, debug probes and the many `generate_dataset_*` /
`gerar_dataset_*` iterations — exists only here.

Expect hard-coded paths (`/home/ericyos/airsim`), the WSL host IP of the
simulator, flat imports between sibling files, and Portuguese comments.

Not included: the data (≈119 GB of datasets, recorded sessions, detection dumps,
trained weights and images, see the top-level README) and a `package.json` /
`pnpm-lock.yaml` pair that belongs to a CLI tool installation, not to this project.
