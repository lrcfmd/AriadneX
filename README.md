# AriadneX

**Unknown Crystalline Phase Discovery from Anomaly-aware Unmixing of Powder X-Ray Diffraction Patterns**

AriadneX builds a diffraction basis from crystal structures, unmixes measured
powder X-ray diffraction (PXRD) patterns, and identifies unexplained signal.
The basis-library encoder is our PXRD-patch transformer.

For the paper examples, prepare your local CIFs using the supplied entry lists
for Pearson's Crystal Data (PCD) and the Inorganic Crystal Structure Database
(ICSD). The releases provide model weights, background-subtracted PXRD patterns,
instrument files, and supporting tables.

Choose [a paper example](#prepare-and-run-a-paper-example) to reproduce the
analysis, or [your own data](#your-own-data) for a general analysis.

## Setup

Use Linux x86_64 with [Git](https://git-scm.com/downloads/linux),
[Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html),
and the [GitHub CLI](https://github.com/cli/cli/blob/trunk/docs/install_linux.md).
Use a GitHub account with access to this repository and its releases.

```bash
gh auth login
git clone https://github.com/lrcfmd/AriadneX.git
cd AriadneX
./scripts/bootstrap.sh
```

Run all remaining commands from this project folder. The setup script installs
the dependencies in the `ariadnex` Conda environment; wait for `Environment ready.`.
The `workflow` command selects this environment automatically.

Set up a [Gurobi licence](https://support.gurobi.com/hc/en-us/articles/360040541251-How-do-I-obtain-a-free-academic-license)
for the machine that will run the analysis.

## Prepare and run a paper example

### 1. Download the model and experimental data

```bash
./scripts/stage_data.sh
```

This downloads the model and both experimental datasets into the project.
The data bundle contains measured patterns, instrument files, and the
`samples.csv`, `ground_truth.csv`, and `basis_order.csv` tables for each example.
Patterns use the paper's sample numbers, such as `sample_9.ASC`.

<details>
<summary>Manual downloads</summary>

Save the files at these paths, relative to the project folder:

| Download | Save as |
| --- | --- |
| [Model weights](https://github.com/lrcfmd/AriadneX/releases/download/checkpoint-v1/last.ckpt) (about 820 MB) | `models/last.ckpt` |
| [Experimental data](https://github.com/lrcfmd/AriadneX/releases/download/example-data-v5/example-data.tar.gz) | `example-data.tar.gz` |

Create `models/` if needed. Extract the experimental data from the project folder:

```bash
tar -xzf example-data.tar.gz
```

</details>

### 2. Get the required CIFs

The `database_id` column in each list gives the entries to obtain:

| Example | Database | Required entries |
| --- | --- | --- |
| Controlled leave-one-out | PCD | [4 entries](examples/controlled_leave_one_out/entries.csv) |
| Open-world phase discovery | ICSD | [874 entries](examples/open_world_phase_discovery/entries.csv) |

Put the downloaded CIFs in a local folder, or use a folder containing your
existing database exports. Choose any folder name and keep the original filenames.
The command matches the required entries using the database IDs inside your CIFs,
including files in subfolders.

### 3. Prepare and run the selected example

Replace the paths below with your CIF folder. `prepare-example` creates
`raw_phases.csv`, then `run` starts the analysis.

**Controlled leave-one-out:** 25 CaCO₃–TiO₂–SnO₂–ZnO mixtures, with each of the
four phases held out in turn. The entry list assigns the A–D labels automatically.

```bash
./workflow prepare-example controlled_leave_one_out --cif-dir "/path/to/your_pcd_cifs"
./workflow run configs/controlled_leave_one_out.yaml
```

**Open-world phase discovery:** 31 measured Ba–Y–Si–Ru–O patterns, numbered as
in the paper.

```bash
./workflow prepare-example open_world_phase_discovery --cif-dir "/path/to/your_icsd_cifs"
./workflow run configs/open_world_phase_discovery.yaml
```

## Your own data

Use `configs/custom.yaml` with your CIF folder, measured PXRD folder, and
GSAS-II instrument file. Follow the [custom data guide](docs/custom_data.md)
for input paths, supported formats, the analysis range, and the optional
sample-composition CSV for element priors.

Download the model:

```bash
./scripts/stage_data.sh --model-only
```

After preparing your files and configuration, run:

```bash
./workflow run configs/custom.yaml
```

## Results and reruns

Progress is shown by stage and sample. When `[workflow] reports: complete`
appears, go to the printed results folder.

Open `04_reports/index.html` inside that folder. It links to sample plots,
phase recovery, anomaly rankings, and CSV tables. Keep the whole `04_reports/`
folder together when copying it. Full reconstructed curves are available in
`anomaly_profiles.npz`.

Running the same command again reuses completed stages. Add `--force` to
recompute, for example:

```bash
./workflow run configs/controlled_leave_one_out.yaml --force
```

To regenerate reports, replace `/path/to/your_run_folder` with the printed results folder:

```bash
./workflow report "/path/to/your_run_folder" --force
```

## Update an existing installation

```bash
git pull
./scripts/bootstrap.sh
./scripts/stage_data.sh --refresh-examples
```

Bootstrap updates the project environment alongside the code.
Refreshing the experimental data backs up the previous example folders.
Then prepare your CIFs again and rerun the chosen example with `--force`, for example:

```bash
./workflow run configs/open_world_phase_discovery.yaml --force
```

For command options, run `./workflow --help`.

## License

AriadneX is released under the [MIT License](LICENSE). The Transformer components
in `src/ariadnex/library/pxrd_encoder.py` are adapted from the MIT-licensed
[Uni-Core](https://github.com/dptech-corp/Uni-Core) library.
