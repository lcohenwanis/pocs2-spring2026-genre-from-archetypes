"""
prepare_training_data.py
------------------------
Build story-level training DataFrames for genre (Comedy vs Drama) classification.

The core representation is the continuous SVD archetype space — the same data
used in the KS analysis in archetype_figures_with_notes.ipynb. Rather than
pooling all comedy characters together (as the KS analysis does), we aggregate
to the story level: for each story, compute the mean and std of each axis score
across its cast.

This gives one row per story with 12 features (mean + std × 6 axes) and a
binary label (is_comedy), suitable for training a classifier.

Main function
-------------
    build_story_features(mat, chars_imdb, n_axes, include_std)
        → DataFrame: 283 stories × 14 columns (12 features + is_comedy + cast_size)

Helper functions (prefixed with _ for internal use)
----------------------------------------------------
    _get_archetype_space(mat)     → (464, 2000) ndarray
    _get_story_index(mat)         → dict {story_name: char_indices (0-based)}
    _get_genre_labels(chars_imdb) → Series {story_name: is_comedy (0 or 1)}
"""

import numpy as np
import pandas as pd
from pathlib import Path

_REPO_ROOT        = Path(__file__).resolve().parents[2]
HAND_LABELS_PATH  = _REPO_ROOT / "prepared_data" / "hand_labels.csv"

# Axis names for the 6 named SVD framework dimensions.
# Sign convention: positive score → left pole, negative → right pole.
#   e.g. fool_hero: high positive = Fool-like, high negative = Hero-like
AXIS_NAMES = [
    "fool_hero",
    "angel_demon",
    "trad_adventurer",
    "lone_wolf_diva",
    "outcast_soph",
    "brute_geek",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_archetype_space(mat: dict) -> np.ndarray:
    """
    Return the (464, 2000) continuous archetype space matrix.

    This is the precomputed equivalent of U_matrix_T @ raw_A from the
    analysis notebooks — each column is a character's projection onto all
    464 SVD axes. The first 6 rows are the named framework axes (AXIS_NAMES).

    Stored in the .mat file as data_archetype_space['character_component_norms'].
    """
    return mat["data_archetype_space"]["character_component_norms"]


def _get_story_index(mat: dict) -> dict:
    """
    Return a dict mapping each story name to a 0-based numpy array of
    character indices.

    Source: mat['data_stories']['storyverses'] and ['storycharacterindices'].
    The .mat file stores indices as 1-based integers; we subtract 1 here so
    they can be used directly for numpy column indexing.
    """
    ds = mat["data_stories"]
    story_names = ds["storyverses"]                   # (341,) array of strings
    story_char_idx = ds["storycharacterindices"]      # (341,) array of 1-based index arrays

    return {
        story_names[i]: (story_char_idx[i] - 1)      # convert to 0-based
        for i in range(len(story_names))
    }


def _get_genre_labels(
    chars_imdb: pd.DataFrame,
    hand_labels_path: "Path | None" = None,
) -> pd.Series:
    """
    Return a Series mapping story_name → is_comedy (1 = Comedy, 0 = Drama).

    Base labels come from IMDB genre tags via XOR filter (stories tagged as
    exclusively Comedy or exclusively Drama). Stories tagged as both are
    excluded unless a hand-labels file is provided.

    Parameters
    ----------
    chars_imdb : pd.DataFrame
        Characters with IMDB metadata.
    hand_labels_path : Path or None
        Path to a CSV with columns [story_name, imdb_genres, label] where
        label is 1 (comedy) or 0 (drama). Rows with blank/NaN labels are
        skipped. When provided, these labels are merged in, replacing any
        IMDB-derived label for the same story.

        Generate the template with:
            python -c "
            from prepare_training_data import generate_hand_label_template
            generate_hand_label_template()
            "
    """
    genres    = chars_imdb.groupby("story_name")["imdb_genres"].first().fillna("")
    is_comedy = genres.str.contains("Comedy")
    is_drama  = genres.str.contains("Drama")
    pure      = is_comedy ^ is_drama
    labels    = is_comedy[pure].astype(int)

    if hand_labels_path is not None and Path(hand_labels_path).exists():
        hl = pd.read_csv(hand_labels_path)
        hl = hl.dropna(subset=["label"])
        hl = hl[hl["label"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
        hl["label"] = hl["label"].astype(float).astype(int)
        hl = hl.set_index("story_name")["label"]
        # Merge: hand labels take precedence, then fill with IMDB-derived labels
        labels = hl.combine_first(labels).astype(int)

    return labels


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_story_features(
    mat: dict,
    chars_imdb: pd.DataFrame,
    n_axes: int = 6,
    include_std: bool = True,
    hand_labels_path: "Path | None" = HAND_LABELS_PATH,
) -> pd.DataFrame:
    """
    Aggregate the continuous SVD archetype scores to the story level.

    For each story in the Comedy/Drama labeled set, extracts all characters'
    scores on the first n_axes SVD dimensions and computes the mean (and
    optionally std) across the cast.

    Parameters
    ----------
    mat : dict
        Loaded .mat file (from data_loader.load_mat()).
    chars_imdb : pd.DataFrame
        Characters with IMDB metadata (from data_loader.load_characters_with_imdb()).
    n_axes : int
        Number of SVD axes to use. Default 6 (the 6 named framework axes).
        Must be <= 464.
    include_std : bool
        If True, include per-axis std across the cast as additional features.
        For single-character stories, std is 0.

    Returns
    -------
    pd.DataFrame
        Index: story_name
        Feature columns (if n_axes=6, include_std=True):
            fool_hero_mean, fool_hero_std,
            angel_demon_mean, angel_demon_std,
            trad_adventurer_mean, trad_adventurer_std,
            lone_wolf_diva_mean, lone_wolf_diva_std,
            outcast_soph_mean, outcast_soph_std,
            brute_geek_mean, brute_geek_std
        Label column:
            is_comedy  (1 = Comedy, 0 = Drama)
    hand_labels_path : Path or None
        Path to hand-labels CSV (see _get_genre_labels). Defaults to
        HAND_LABELS_PATH (prepared_data/hand_labels.csv). Set to None
        to use IMDB-only labels (283 stories).

    Returns
    -------
    pd.DataFrame
        Metadata column:
            cast_size  — number of characters in the story's cast.
                         Use as sample_weight when fitting sklearn models to
                         down-weight stories whose features are estimated from
                         only 1–2 characters.

    Notes
    -----
    Sign convention for axis scores:
        positive → left pole (Fool, Angel, Traditionalist, Lone Wolf, Outcast, Brute)
        negative → right pole (Hero, Demon, Adventurer, Diva, Sophisticate, Geek)

    This means a story with a high mean fool_hero score has a cast that leans
    Fool-like; a story with a strongly negative mean leans Hero-like.
    """
    archetype_space = _get_archetype_space(mat)   # (464, 2000)
    story_index     = _get_story_index(mat)        # {story_name: 0-based indices}
    genre_labels    = _get_genre_labels(chars_imdb, hand_labels_path)

    axis_labels = AXIS_NAMES[:n_axes]

    rows = []
    for story_name, is_comedy in genre_labels.items():
        if story_name not in story_index:
            continue

        char_idx = np.atleast_1d(story_index[story_name])     # 0-based; scalar → array
        scores   = archetype_space[:n_axes, char_idx]         # (n_axes, n_chars)

        n_chars = len(char_idx)
        row = {"story_name": story_name, "is_comedy": int(is_comedy), "cast_size": n_chars}

        for j, axis in enumerate(axis_labels):
            row[f"{axis}_mean"] = scores[j].mean()
            if include_std:
                # ddof=0: summarising the actual cast, not estimating a population
                row[f"{axis}_std"] = scores[j].std(ddof=0)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("story_name")

    # Put cast_size and is_comedy last for readability
    meta = [c for c in ["cast_size", "is_comedy"] if c in df.columns]
    feat = [c for c in df.columns if c not in meta]
    return df[feat + meta]
