# Presentation Notes
## Genre in the Archetype Space: Character Clustering and Genre Prediction

---

## The Elevator Pitch

"We took 2,000 fictional characters — everyone from Walter White to Pinkie Pie — rated on 464 personality trait scales, and asked: does genre leave a fingerprint on who the characters are? We find that it does. Comedy and drama casts occupy measurably different regions of personality space, and a simple classifier trained on that structure predicts genre substantially better than chance."

---

## The Data and Shared Space (top center of poster)

**What the dataset is:**
- 2,000 fictional characters rated by crowd workers on 464 bipolar trait scales (e.g., "goofy :: unfrivolous", "chatty :: reserved")
- Each trait is a slider between two poles — a character gets a score on every one of the 464 scales
- This is the Archetypometrics dataset — think of it as a personality genome for fictional characters

**The 6 archetype axes:**
- SVD (Singular Value Decomposition) compresses 464 trait dimensions down to 6 primary axes that explain the most variance
- Think of SVD like finding the 6 most important "personality factors" that summarize all 464 traits
- Each axis is bipolar: Fool ↔ Hero, Angel ↔ Demon, Traditionalist ↔ Adventurer, Lone Wolf ↔ Diva, Outcast ↔ Sophisticate, Brute ↔ Geek
- Every character becomes a point in this 6-dimensional space

**How stories enter the picture:**
- 341 stories matched to IMDB genre labels → 247 after filtering to pure Comedy or pure Drama (no mixed-genre stories)
- 87 Comedy (35%), 160 Drama (65%) — class imbalance is real and matters for the classifier
- Mean cast size: 5.7 characters (SD 3.75); 19% of stories have fewer than 3 rated characters

---

## Data Preprocessing (how features are built)

**Character level → Story level:**
1. For each story, take all rated characters and compute the mean score on each of the 6 archetype axes → 6 features per story
2. Also compute cast size (number of rated characters)
3. Compute cast gyration (see below) → 1 more feature
4. Final feature set: 8 features per story

**What cast gyration is:**
- Think of the cast as a cloud of points in 6D archetype space
- Find the centroid (center of mass) of that cloud — the "average character" for that story
- Gyration measures how spread out the characters are around that centroid
- High gyration = diverse cast with characters spread across archetype space
- Low gyration = homogeneous cast where everyone is similar
- Formula: square root of the mean squared distance of each character from the cast centroid

**Why gyration instead of per-axis standard deviation:**
- Per-axis std measures spread on each axis independently (6 separate numbers)
- Gyration measures total spread across all 6 dimensions at once (1 number)
- It captures the full geometry of the cast, not just axis-by-axis variance
- Using gyration in place of the 6 std features improved AUC from 0.724 → 0.746

**Class imbalance handling:**
- 160 dramas vs. 87 comedies — if you just predict "drama" every time, you get 65% accuracy
- Solution: within each training fold, fit a Gaussian distribution to the comedy training examples and sample synthetic comedy stories to balance the classes
- This is done inside cross-validation so test data is never touched during oversampling
- Effect: F1 improved from 0.543 → 0.617

---

## Research Question 1: How do characters cluster in archetype space by genre?

### Figure: Top 20 Traits Bar Chart (left side of RQ1)

**How to read it:**
- Each bar is one of the 464 bipolar trait pairs (e.g., "goofy :: unfrivolous")
- Bar length = mean difference in trait score between comedy and drama characters
- Green bars extend right = comedy characters score higher on that trait
- Purple bars extend left = drama characters score higher on that trait
- Sorted by absolute difference — the traits that separate genres most are at the top

**What it tells you:**
- Comedy characters are more: coordinated, open-book, chatty, playful, funny, cheery, sunny
- Drama characters are more: unfrivolous, dramatic, serious, sorrowful, haunted, stoic
- 304 of 464 traits (65%) are statistically significant at p < 0.05 — genre is not a subtle signal at the trait level
- The top trait pair "unfrivolous :: goofy" has a mean difference of ~0.13, which is large given these scores are on a standardized scale

**What to say to an audience:**
"This is 464 personality dimensions compressed to the 20 that matter most. Comedy characters are funnier, chattier, more playful. Drama characters are more serious, more haunted, more stoic. None of that is surprising — but the fact that it shows up this cleanly in crowdsourced personality ratings is the point."

---

### Figures A–F: Archetype Distribution Comparisons (right side of RQ1)

**How to read them:**
- Each panel shows one of the 6 archetype axes
- Green bars above the center line = comedy character density
- Purple bars below the center line = drama character density (mirrored downward)
- The axis label shows which archetype is on which end (e.g., "← Fools | Heroes →")
- Triangles mark the median for each genre; KS statistic measures distributional separation

**Primary archetypes (A, B, C) — the most informative:**

- **Panel A (Fool ↔ Hero):** Drama skews Hero (median 1.79), comedy skews slightly less Hero (median 1.21). KS = 0.16, p < 10⁻⁴. Both genres lean Hero but drama leans harder. Almost no pure Fools in either genre.
- **Panel B (Angel ↔ Demon):** Comedy skews Angel (median -0.44), drama is near neutral (median 0.09). KS = 0.10, p = 0.007. Comedy casts are slightly more angelic on average.
- **Panel C (Traditionalist ↔ Adventurer):** Comedy skews Adventurer (median 0.83), drama is near neutral (median 0.26). KS = 0.14, p < 10⁻⁴. This is the cleanest visual separation of the primary axes.

**Secondary archetypes (D, E, F) — weaker signal:**

- **Panel D (Lone Wolf ↔ Diva):** Comedy skews Diva (median 0.52), drama skews Lone Wolf (median 0.31). KS = 0.11, p = 0.002.
- **Panel E (Outcast ↔ Sophisticate):** No meaningful separation. KS = 0.05, p = 0.333. Genre doesn't care about this axis.
- **Panel F (Brute ↔ Geek):** No meaningful separation. KS = 0.03, p = 0.895. Genre doesn't care about this axis either.

**What to say to an audience:**
"Think of each panel as asking: does genre shape which archetypes populate a story? For the Traditionalist-Adventurer axis and the Fool-Hero axis, the answer is clearly yes — comedy casts skew Adventurer, drama casts skew Hero. Two of the six axes (E and F) show essentially no separation at all. Genre is selective about which personality dimensions it touches."

**The cast diversity finding:**
- Comedy casts have a higher radius of gyration than drama casts (4.37 vs. 3.98)
- Comedy casts are more spread out in archetype space — they mix archetypes more freely
- Drama casts are more internally consistent — characters cluster closer together in personality space
- This finding — cast diversity as a property of comedy — gets repurposed as a predictor in RQ2

---

## Research Question 2: Can we predict genre from archetype composition?

### Figure: Cast Size Distribution by Genre

**How to read it:**
- Grouped bar chart showing how many stories of each genre fall into cast size bins
- X-axis = number of rated characters in the cast; Y-axis = number of stories

**What it tells you:**
- Both genres have many thin-cast stories (cast size 1–4)
- Drama has more stories with larger casts (5+)
- The distributions overlap heavily — cast size alone is not a clean separator (KS = 0.10, p = 0.574, not significant)
- The thin-cast problem: ~40% of stories have fewer than 5 rated characters, which makes gyration unreliable for those stories

---

### Figure: Stories in Archetype Space (scatter plot)

**How to read it:**
- Each dot is one story, positioned by its cast's average archetype scores
- X-axis: average of all characters on the Traditionalist ↔ Adventurer axis
- Y-axis: average of all characters on the Fool ↔ Hero axis
- Green = Comedy, Purple = Drama
- Stars = genre centroids (center of mass of all stories of that genre)

**What it tells you:**
- Comedy stories (green) drift right — their casts lean Adventurer on average
- Drama stories (purple) cluster more toward the left — their casts lean Traditionalist
- Both genres sit above zero on the Hero axis — neither genre produces many Fool-dominant casts
- The drama centroid (purple star) is higher and further left than the comedy centroid (green star)
- There is substantial overlap in the middle — genre is not perfectly determined by these two dimensions alone

**The connection to RQ1:**
- These are the exact same two axes that showed the strongest character-level separation in panels A and C
- The separation that appeared in individual character distributions in RQ1 holds when you aggregate to the story level
- This is the bridge between the two research questions: character-level signal → story-level signal → predictive signal

**What to say to an audience:**
"Each dot is a story, placed in personality space based on who its characters are. You can see comedies drifting toward the Adventurer end, dramas clustering toward the Traditionalist end. The stars mark where a typical story of each genre lives. There's a lot of overlap in the middle — which is exactly why our AUC is 0.745 and not 0.95. But the separation is real, and these are the same two axes that separated individual characters in the histograms on the left."

---

### Figure: ROC Curve

**How to read it:**
- X-axis = False Positive Rate (dramas incorrectly called comedy)
- Y-axis = True Positive Rate (comedies correctly identified)
- The diagonal dashed line = chance (AUC = 0.5, a coin flip)
- Individual fold lines (light purple) show the 5-fold CV variation
- Green line = mean ROC curve across all folds; shaded band = ±1 SD

**What it tells you:**
- AUC = 0.745 ± 0.038 — the model correctly ranks a comedy above a drama 74.5% of the time
- The curve consistently sits above the diagonal across all 5 folds — the result is stable
- AUC is the right metric here given class imbalance; raw accuracy would be misleading

**What to say to an audience:**
"This curve asks: if I give the model a comedy and a drama and ask which is which, how often does it get it right? The answer is about 74.5% of the time, versus 50% for a coin flip. The fact that it does this using only the personality profile of the cast — no plot, no dialogue, no title — is the interesting part."

---

### Figure: Precision-Recall Curve

**How to read it:**
- X-axis = Recall (what fraction of all comedies did the model find?)
- Y-axis = Precision (of the stories the model called comedy, what fraction actually were?)
- Dashed horizontal line = no-skill baseline (the comedy class prevalence, ~35%)
- Green line = mean PR curve; shaded band = ±1 SD

**What it tells you:**
- AP (average percision) = 0.636 ± 0.062 vs. 0.35 baseline — substantial improvement over chance
- As you ask the model to find more comedies (higher recall), precision drops — there's a tradeoff
- The curve staying well above the baseline across the full recall range shows the model is genuinely informative, not just lucky at one operating point

**What to say to an audience:**
"This is the honest view of prediction given our class imbalance. Even when we ask the model to cast a wide net and find most comedies, it stays meaningfully above the baseline precision. The baseline here is 35% — just the fraction of comedies in the dataset. Our model stays well above that across the full range."

---

### Figure: Confusion Matrix

**How to read it:**
- Rows = actual genre; columns = predicted genre
- Top-left (60): comedies correctly called comedy (true positives)
- Top-right (27): comedies incorrectly called drama (false negatives)
- Bottom-left (48): dramas incorrectly called comedy (false positives)
- Bottom-right (112): dramas correctly called drama (true negatives)

**What it tells you:**
- The model is better at identifying dramas (112/160 = 70%) than comedies (60/87 = 69%)
- 48 dramas are called comedy — the model over-predicts comedy somewhat
- Precision = 0.556, Recall = 0.690, F1 = 0.617, Accuracy = 0.697
- Majority baseline accuracy = 0.648 (just predicting "drama" every time)
- The model beats the baseline but the margin is honest — not a clean separation

---

## Key Findings Summary

**RQ1:**
- Drama casts lean Hero, Traditionalist, Lone Wolf
- Comedy casts lean Fool, Adventurer, Diva
- Comedy casts are significantly more diverse in archetype space (gyration 4.37 vs. 3.98)
- 65% of 464 individual traits distinguish genres at p < 0.05

**RQ2:**
- AUC = 0.745 vs. 0.5 chance; accuracy 0.697 vs. 0.648 majority baseline
- The two axes that most distinguish genres (Trad-Adventurer, Fool-Hero) are also the top two predictors in the logistic regression
- Replacing per-axis std with gyration improves AUC (0.724 → 0.746) — cast diversity is more informative than per-axis spread
- Bootstrap oversampling improves F1 substantially (0.543 → 0.617)

---

## Limitations (be ready for these questions)

- **Thin casts:** 40% of stories have fewer than 5 rated characters. Gyration is unreliable for these stories, and cast size may be acting as a proxy for this unreliability rather than a genuine genre signal.
- **Class imbalance:** 160 dramas vs. 87 comedies. Gaussian oversampling is a reasonable fix but it propagates noise from thin-cast stories and the comedy covariance is estimated from only ~70 examples per fold.
- **2D scatter is a projection:** The scatter plot shows only 2 of the 8 features. The actual classifier operates in 8D — the visual separation in the scatter understates what the model sees.
- **Genre labels:** IMDB labels are noisy. "Pure comedy" and "pure drama" filtering removes mixed-genre stories but the remaining labels aren't ground truth.
- **Linear model ceiling:** Logistic regression can only draw a straight hyperplane. A non-linear model might do better, but with 247 stories and 8 features there isn't much room to push before overfitting.
