# Your own data

Complete the [setup](../README.md#setup) first. Run the commands below from the
project folder and download the model:

```bash
./scripts/stage_data.sh --model-only
```

Use `configs/custom.yaml` with your original CIF filenames. AriadneX discovers
the input files and assigns phase IDs automatically.

1. Create two folders:

   ```bash
   mkdir -p data/custom/cifs data/custom/patterns
   ```

2. Put your original `.cif` files in `data/custom/cifs/`, and your measured PXRD
   files in `data/custom/patterns/`. Keep their original filenames. Each PXRD
   file contains two-theta in degrees and background-subtracted intensity;
   supported formats are ASC, XY, XYE, TXT, CSV, NPY and NPZ. Use a different
   filename stem for each measurement, for example `sample_1.xy` and
   `sample_2.xy`.
   In text files, the first two columns are two-theta and intensity; an XYE
   uncertainty column is ignored. CSV files may include a header row.

3. Save your measurement's GSAS-II instrument parameters as
   `data/custom/experimental.instprm`. These parameters must describe your
   measurement, including its wavelength and peak broadening.

4. Open `configs/custom.yaml`. Set `analysis.theta_min` and
   `analysis.theta_max` to the two-theta range you want to analyse. The defaults
   are 15–75°. AriadneX uses measured points inside this range and tells you if a
   scan covers less. If your files are elsewhere, change the three paths under
   `inputs`.

5. Run:

   ```bash
   ./workflow run configs/custom.yaml
   ```

AriadneX automatically checks the CIFs, builds and reduces the basis library,
simulates the structures, unmixes the PXRD patterns, and generates the report.
It keeps the original CIFs unchanged and records the generated phase IDs,
source paths, filtering decisions, and representatives in
`01_library/manifest.csv` and `01_library/representatives.csv`.

When the run finishes, open `04_reports/index.html` in the printed results
folder.

## Optional element prior

Custom runs leave the element prior off by default. If you know each sample's
nominal composition, you can use it to exclude candidate phases containing
elements known to be absent from that sample.

1. Create `data/custom/compositions.csv` with one row per measured sample:

   ```csv
   sample_id,nominal_composition
   sample_1,Ba0.092Y0.408Ru0.414Si0.086O1.704
   sample_2,Ba0.5Si0.5O1.5
   ```

   Match each `sample_id` to its pattern filename without the extension, such
   as `sample_1.xy`. Enter the nominal composition of the whole sample,
   replacing the example compositions with your own.

2. Add this block to `configs/custom.yaml`:

   ```yaml
   unmix:
     element_prior:
       enabled: true
       considered_elements: [Ba, Y, Ru, Si]
       sample_description_path: ../data/custom/compositions.csv
   ```

   Set `considered_elements` to the elements whose presence or absence you
   want to constrain in your dataset. Within that list, an omitted element or
   a zero amount is treated as absent. For example, the second row excludes
   candidates containing Y or Ru. Elements outside the list are not excluded
   by this check. Leave the prior off if this information is unknown.

Keep `inputs.mixtures` pointing to the pattern folder and run the same custom
command. Every sample needs a matching composition row when the prior is on.
The prior uses element presence rather than enforcing composition ratios;
with `max_base: auto`, it also sets the phase-count limit to the number of
present elements in `considered_elements`.
