# Genre from Archetypes: Distinguishing Comedy from Drama via Character Trait Distributions

Can the character traits and archetype compositions of a story's cast distinguish comedies from dramas? Using the Archetypometrics dataset (Dodds et al.), we compare comedy and drama character populations along 464 bipolar trait dimensions and a six-dimensional archetype space, then build a genre classifier trained purely on cast-level personality features.

---

## Research Questions

**RQ1:** Do comedy and drama characters occupy different regions of archetype and trait space?

**RQ2:** Can genre be predicted from cast archetype composition alone?

---

## Data

**Archetypometrics dataset** — 2,000 fictional characters across 341 stories, each rated by crowd workers on 464 bipolar trait scales (e.g., *goofy :: unfrivolous*, *chatty :: reserved*). A truncated SVD compresses the 464-dimensional trait matrix into 6 archetype axes that explain the most variance:

| Axis | Poles |
|---|---|
| 1 | Fool ↔ Hero |
| 2 | Angel ↔ Demon |
| 3 | Traditionalist ↔ Adventurer |
| 4 | Lone Wolf ↔ Diva |
| 5 | Outcast ↔ Sophisticate |
| 6 | Brute ↔ Geek |

**IMDb genre labels** — matched to story names via a curated mapping. Stories labeled exclusively Comedy or exclusively Drama are retained (the XOR filter), yielding **65 comedies** and **138 dramas**.

---

## Notebooks

| Notebook | What it does |
|---|---|
| [join_datasets.ipynb](notebooks/join_datasets.ipynb) | Joins the Archetypometrics character data with IMDb genre labels; produces `characters_with_imdb.csv` and `story_imdb_mapping.csv` |
| [prepare_data.ipynb](notebooks/prepare_data.ipynb) | Builds the story-level feature matrix (`df_model.csv`): mean archetype scores per story, cast size, and cast radius of gyration |
| [initial_archetype_eda.ipynb](notebooks/initial_archetype_eda.ipynb) | Early exploratory analysis of archetype distributions and IMDB genre coverage |
| [archetype_figures_comedy_vs_drama.ipynb](notebooks/archetype_figures_comedy_vs_drama.ipynb) | Main analysis notebook: KS tests across all 6 archetype dimensions and all 464 traits, bootstrapped radii of gyration, mirrored histogram figures, trait KDE plots |
| [archetype_figures_comedy_vs_drama_XOR.ipynb](notebooks/archetype_figures_comedy_vs_drama_XOR.ipynb) | Same analysis restricted to stories that are *exclusively* comedy or drama (no mixed-genre stories); this is the primary dataset used in the paper |
| [prediction.ipynb](notebooks/prediction.ipynb) | Logistic regression classifier using the 6 mean archetype scores + cast size; 5-fold CV with bootstrap oversampling to handle class imbalance |
| [prediction_gyration.ipynb](notebooks/prediction_gyration.ipynb) | Replaces per-axis std features with cast radius of gyration; improves AUC from 0.724 → 0.746 |
| [prediction_all_traits.ipynb](notebooks/prediction_all_traits.ipynb) | Classifier trained on all 464 raw trait means rather than compressed archetype scores |

---

## Key Findings

### Archetype-level differences (RQ1)

Four of six archetype axes show statistically significant distributional differences (KS test):

- **Fool ↔ Hero** (KS = 0.17, p < 0.0001): drama characters lean strongly Hero; comedy characters are closer to the Fool pole.
- **Traditionalist ↔ Adventurer** (KS = 0.16, p < 0.0001): comedy casts skew Adventurer; drama casts skew Traditionalist.
- **Lone Wolf ↔ Diva** (KS = 0.14, p = 0.002): comedy characters skew Diva (more expressive, socially prominent).
- **Angel ↔ Demon**, **Outcast ↔ Sophisticate**, **Brute ↔ Geek**: no significant separation.

Comedy casts also have a higher bootstrapped radius of gyration (4.37 vs. 3.98), meaning comedy casts are more diverse in archetype space — they mix character types more freely than drama casts.

### Trait-level differences (RQ1)

Of 464 bipolar traits, **304 (65%)** show statistically significant differences between comedy and drama characters (p < 0.05). The top separating traits:

- Comedy characters are more: **goofy**, **open-book**, **weird**, **happy**, **blissful**, **jovial**, **chatty**, **awkward**, **expressive**
- Drama characters are more: **unfrivolous**, **secretive**, **profound**, **deep**, **stoic**, **haunted**, **confidential**

### Genre prediction (RQ2)

A logistic regression trained on 8 features (6 mean archetype scores + cast size + radius of gyration), evaluated via 5-fold cross-validation with Gaussian bootstrap oversampling for class balance:

| Metric | Value |
|---|---|
| AUC | 0.745 ± 0.038 |
| Accuracy | 0.697 (majority baseline: 0.648) |
| F1 (comedy) | 0.617 |
| Average precision | 0.636 (baseline: 0.35) |

The Fool–Hero and Traditionalist–Adventurer axes are the top two predictors in the regression — the same axes that most separate individual characters in RQ1.

---

## Repository Structure

```
datasets/
  archetypometricsdata2000.mat       # Raw Archetypometrics data (MATLAB format)
  data/plain/current/2000/           # Archetypometrics plain-text exports
  IMDBdata - download 2_19_26/       # IMDb TSV files (title.basics, title.principals, etc.)
  imdb_archetype_character_comedy_drama.csv
  story_character_summary.csv        # Story–genre mapping used for label assignment

prepared_data/
  characters_with_imdb.csv           # Characters joined with IMDb genre labels
  story_imdb_mapping.csv             # Story name → IMDb tconst mapping
  story_aggregated_data.csv          # Story-level archetype means
  df_model.csv                       # Final feature matrix for classification
  hand_labels.csv                    # Manually verified genre labels

figures/                             # Output figures (PDF + PNG)
  xor-figures/                       # Figures from the XOR-filtered dataset (used in paper)

notebooks/                           # Analysis notebooks (see table above)

paper_sections.md                    # Full paper draft
presentation.md                      # Poster/presentation speaker notes
```

---

## Reference

Dodds et al., *Archetypometrics: A data-driven archetype space for fictional characters* (dataset).
