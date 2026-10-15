The files in this directory are results from extracting variables for 10 randomly chosen public ESS-DIVE datasets submitted within the three years before 2026-10-15, as candidate material for improving BERVO.

## Selection

`select_datasets.py` chose the datasets (record of the draw: `selection.json`):

- Catalogue: all 1588 public ESS-DIVE packages as of 2026-10-15.
- Eligible: `dateUploaded` on or after 2023-10-15 **and** `datePublished` year 2023 or later (617 packages). ESS-DIVE does not record a first-submission date: `dateUploaded` is the latest version's upload, and `datePublished` is a year only. Requiring both keeps out older datasets that were only revised recently. The API's `beginDate` filter was not used; it does not filter on submission date.
- Draw: eligible packages sorted by id, shuffled with seed `20261015`, and taken in order. Two were skipped because they had no file TROWEL can read (.csv, .tsv, .txt, .xls, .xlsx, .xml): doi:10.15485/3487885 and doi:10.15485/3098419.

The catalogue changes over time, so rerunning the script later may give a different set.

## DOIs

| DOI | Variables | Dataset |
|---|---|---|
| doi:10.15485/2395957 | 5 | Active Layer Depth and Permafrost Temperatures at the Teller 47 Field Site, Seward Peninsula, Alaska, 2022 |
| doi:10.15485/3412618 | 22 | X-ray Absorption Spectroscopy and Neutron Scattering Data, Mineral Incubation Experiment, Walker Branch Watershed, TN (2020 - 2021) |
| doi:10.15485/3015733 | 111 | Dataset for Cruz-O'Byrne et al (2026): "Divergent biogeochemical responses in upland coastal forest soils to repeated flooding and shifts in water chemistry" |
| doi:10.15485/2475392 | 14 | Leaf mass area, leaf carbon and nitrogen content of 18 plant species, Seward Peninsula, Alaska, 2023 |
| doi:10.15485/3025240 | 0 | Meteoric 10Be Flux Calibration Data for the East River Watershed, Colorado, USA |
| doi:10.15485/2589266 | 1* | Integrating Characteristic Arctic Vegetation in a Land Surface Model Improves Representation of Carbon Dynamics Across a Tundra Landscape: Modeling Archive |
| doi:10.15485/2998439 | 21 | Porewater chemistry in Typha-dominated brackish tidal marsh, PIE LTER, Plum Island Sound, MA, July 2022–September 2024 |
| doi:10.15485/2324642 | 0 | Model output from simulations of manganese-carbon interactions in temperate forest soil profiles |
| doi:10.5440/1975126 | 0 | Vegetation Warming Experiment: Landscape-scale digital camera imagery for vegetation phenology, Utqiagvik (Barrow), Alaska, 2022 |
| doi:10.15485/3020155 | 19 | Fungal mat growth and leaf colonization at the TRACE warming experiment, Mar - Aug 2024, Luquillo, Puerto Rico |

Counts are per dataset; a name shared by several datasets is counted once in each. The datasets with 0 hold their data in zip archives or images; their only readable files are READMEs, FLMD files, or model templates.

## Commands

```bash
trowel get-essdive-metadata -p dois.txt -o .
trowel get-essdive-variables -p filetable.tsv -o .
```

Run with TROWEL at commit 65613da, without an `ESSDIVE_TOKEN`.

## Outputs

- `variable_names.tsv`: 189 unique variable names; 56 have units and definitions from a data dictionary.
- `data_dictionaries.tsv`: 60 data dictionary rows.
- `results.tsv`, `filetable.tsv`, `frequencies.tsv`: dataset metadata, file list, and catalogue-listed variable frequencies.

## Caveats for curation

- *The single name from doi:10.15485/2589266, `196.19585    64.862976     1     1`, is not a variable. `zone_mappings.txt` has no header row, so its first data row was read as column names.
- doi:10.15485/3015733 supplies 111 of the 189 names, so porewater chemistry is overrepresented.
- Many names are summary or replicate columns rather than distinct variables (for example `ph fw mean` / `ph fw std`, `day 28 rep 1` / `day 28 rep 2`, `quartz rep 1`), and many are abbreviations (`hix`, `bix`, `fi`, `suva`, `bw`, `fw`). They need mapping to their base variables before comparison with BERVO.
