"""
data_loader.py
--------------
Single source of truth for loading all project data assets.

All paths resolve relative to the repo root, so functions work regardless of
where a notebook is run from.
"""

import scipy.io
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

MAT_PATH              = _REPO_ROOT / "datasets"      / "archetypometricsdata2000.mat"
IMDB_TSV_PATH         = _REPO_ROOT / "datasets"      / "title.basics.tsv"
CHARACTERS_IMDB_PATH  = _REPO_ROOT / "prepared_data" / "characters_with_imdb.csv"
CHAR_ARCHETYPES_PATH  = _REPO_ROOT / "prepared_data" / "df_char_archetypes.csv"
MODEL_DATA_PATH       = _REPO_ROOT / "prepared_data" / "df_model.csv"


# ---------------------------------------------------------------------------
# Raw .mat file
# ---------------------------------------------------------------------------

def load_mat(path: Path = MAT_PATH) -> dict:
    """
    Load the raw MATLAB file.

    Returns a plain dict with keys:
        data_characters, data_archetype_space, data_raw, data_traits,
        data_stories, framework_archetypes, (and _latex variants)

    simplify_cells=True is required for clean dict-style field access.
    """
    return scipy.io.loadmat(str(path), simplify_cells=True)


# ---------------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------------

def load_characters(mat: dict | None = None) -> pd.DataFrame:
    """
    One row per character (2,000 rows).

    Columns:
        character_name      — string
        story_name          — string (341 unique values)
        primary_archetype   — compound label, e.g. 'Angel-Brute-Hero'

    The .mat field 'character_story_typenames' encodes both as
    'CharName/StoryName'; we split on '/' to isolate the story name.
    """
    if mat is None:
        mat = load_mat()

    chars_raw       = mat["data_characters"]
    archetype_space = mat["data_archetype_space"]
    pairs           = list(chars_raw["character_story_typenames"])

    return pd.DataFrame({
        "character_name":   list(chars_raw["character_typenames"]),
        "story_name":       [p.split("/", 1)[1] if "/" in p else p for p in pairs],
        "primary_archetype": list(archetype_space["character_archetypes_ordered"]),
    })


# ---------------------------------------------------------------------------
# Archetype ratios (232 compound archetypes)
# ---------------------------------------------------------------------------

def load_archetype_ratios(mat: dict | None = None) -> pd.DataFrame:
    """
    2,000 characters × 232 compound archetype ratio scores.

    Each row has exactly one non-zero value — the character's assigned
    compound archetype. Values are unnormalized SVD scores (~0–25).

    Integer RangeIndex (matches character order in load_characters()).
    """
    if mat is None:
        mat = load_mat()

    acs             = mat["data_archetype_space"]
    n_archetypes    = acs["archetype_ratios"].shape[1]
    archetype_names = list(acs["archetypes"][:n_archetypes])

    return pd.DataFrame(acs["archetype_ratios"], columns=archetype_names)


# ---------------------------------------------------------------------------
# Framework component scores (6 named axes + 5 essential traits)
# ---------------------------------------------------------------------------

def load_framework_scores(mat: dict | None = None) -> dict:
    """
    SVD-derived scores for each character on 11 archetype axes.

    Returns a dict with three DataFrames (2,000 rows × 11 columns each)
    and the axis name list:

        component_norms     — raw component magnitude per axis
        variance_explained  — % variance explained per axis
        alignment_cosines   — cosine of alignment; sign = which pole
                              (positive = right pole: Hero/Demon/Adventurer/Diva/Sophisticate/Geek)

        axis_names          — list of 11 strings, e.g. ['Fool_Hero', 'Angel_Demon', ...]

    Named axes (indices 0–5) map to the 6 primary bipolar dimensions.
    Axes 6–10 are unnamed ('Essential_7' … 'Essential_11').
    """
    if mat is None:
        mat = load_mat()

    acs    = mat["data_archetype_space"]
    fw_df  = pd.DataFrame(mat["framework_archetypes"]["names_and_descriptions"])

    axis_names = []
    for i in range(11):
        try:
            n1 = fw_df.loc[i, "name1"]
            n2 = fw_df.loc[i, "name2"]
            if isinstance(n1, str) and n1 and isinstance(n2, str) and n2:
                axis_names.append(f"{n1}_{n2}")
            else:
                axis_names.append(f"Essential_{i + 1}")
        except (KeyError, IndexError):
            axis_names.append(f"Essential_{i + 1}")

    def _to_df(arr):
        # arr is (11, 2000) → transpose to (2000, 11)
        return pd.DataFrame(arr.T, columns=axis_names)

    return {
        "component_norms":    _to_df(acs["character_component_norms"]),
        "variance_explained": _to_df(acs["character_variance_explained"]),
        "alignment_cosines":  _to_df(acs["character_alignment_cosines"]),
        "axis_names":         axis_names,
    }


# ---------------------------------------------------------------------------
# Raw trait matrix (464 bipolar traits)
# ---------------------------------------------------------------------------

def load_traits(mat: dict | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        trait_df   — DataFrame with columns ['trait', 'trait_flip'] (464 rows)
                     'trait'      e.g. 'playful :: serious'
                     'trait_flip' e.g. 'serious :: playful'

        A          — (464, 2000) float array of mean crowd-sourced trait scores
        Acounts    — (464, 2000) float array of rating counts per trait per character
        Astds      — (464, 2000) float array of rating standard deviations

    To get a trait score for character at index i:  A[:, i]
    To get scores for trait at index j across all characters:  A[j, :]
    """
    if mat is None:
        mat = load_mat()

    traits   = mat["data_traits"]
    raw      = mat["data_raw"]

    trait_df = pd.DataFrame({
        "trait":      list(traits["trait_typenames"]),
        "trait_flip": list(traits["trait_flip_typenames"]),
    })

    return trait_df, raw["A"], raw["Acounts"], raw["Astds"]


# ---------------------------------------------------------------------------
# Prepared CSVs
# ---------------------------------------------------------------------------

def load_characters_with_imdb(path: Path = CHARACTERS_IMDB_PATH) -> pd.DataFrame:
    """
    2,000 characters joined to IMDB metadata (95.2% match rate).

    Columns: character_name, story_name, primary_archetype, match_type,
             imdb_tconst, imdb_titleType, imdb_primaryTitle,
             imdb_startYear, imdb_genres
    """
    return pd.read_csv(path)


def load_char_archetypes(path: Path = CHAR_ARCHETYPES_PATH) -> pd.DataFrame:
    """
    2,000 characters × 235 columns.

    Columns: story_name, character_name, primary_archetype,
             <232 compound archetype ratio columns>, Non Archetypes

    Note: not tracked in git — must exist on disk (generated by prepare_data.ipynb).
    """
    return pd.read_csv(path)


def load_model_data(path: Path = MODEL_DATA_PATH) -> pd.DataFrame:
    """
    Story-level model-ready dataset. 283 rows × 7 columns.
    Index: story_name

    Feature columns (proportions summing to 1.0 per row):
        Adventurer, Angel, Demon, Fool, Hero, Traditionalist

    Label:
        is_comedy  — 1 = Comedy, 0 = Drama
        Class distribution: 95 comedy (33.6%), 188 drama (66.4%)

    Note: these are raw proportions. Apply CLR transform before linear models.
    """
    return pd.read_csv(path, index_col="story_name")
