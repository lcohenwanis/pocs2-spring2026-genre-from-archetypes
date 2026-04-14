# Genre from Archetypes: Distinguishing Comedy from Drama via Character Trait Distributions

---

## Introduction

Stories are defined not only by plot but by the characters who inhabit them. Across literary and narrative traditions, recurring character types—archetypes—have long been understood as structural building blocks of storytelling. The fool, the hero, the adventurer, and the geek are recognizable not merely as aesthetic choices but as functional roles that shape how a story feels and what genre it occupies. Yet the connection between character composition and genre has rarely been studied at scale using quantitative methods.

In this work, we ask a simple but precise question: can the character traits and archetype distributions of a story's cast distinguish comedies from dramas? Using the Archetypometrics dataset of Dodds et al., which encodes 2,000 fictional characters across 341 stories along 464 bipolar trait dimensions and a six-dimensional archetype space derived from singular value decomposition (SVD), we compare the character populations of 73 comedy stories (449 characters) and 55 drama stories (330 characters) drawn from film, television, and literature. Genre labels are sourced from IMDb.

Our analysis proceeds at two levels. First, we examine how comedy and drama characters distribute across the six primary archetype dimensions—Fool–Hero, Angel–Demon, Traditionalist–Adventurer, Lone Wolf–Diva, Outcast–Sophisticate, and Brute–Geek—using distributional comparison (Kolmogorov–Smirnov tests) and bootstrapped radii of gyration. Second, we conduct a trait-level analysis across all 464 dimensions to identify the specific character traits that most reliably distinguish genre. We find that genre leaves a strong and interpretable signature in character trait space: comedy characters are significantly goofier, chattier, happier, more awkward, and more expressive than their drama counterparts, while drama characters lean more heroic, secretive, profound, and stoic. These differences are statistically robust and align coherently with the functional demands each genre places on its characters.

---

## Related Works

**Archetype theory and computational character analysis.** The idea that characters in narrative can be reduced to a finite set of recurring types has roots in the structural analyses of Propp (1928), who identified seven character roles in Russian folktales, and Campbell (1949), whose monomyth posited a universal hero archetype. Jung (1959) extended this to a psychological framework of universal archetypes grounded in the collective unconscious. More recently, computational approaches have attempted to operationalize these intuitions. Bamman, Underwood, and Smith (2014) applied topic modeling and co-occurrence networks to identify character types in film; Vala et al. (2015) studied character networks in novels; and Reagan et al. (2016) mapped emotional arcs of stories. The Archetypometrics framework of Dodds et al. advances this tradition by constructing a data-driven archetype space from crowd-sourced character trait ratings, positioning characters in a continuous low-dimensional space without presupposing a fixed typology.

**Trait-based character representation.** Crowdsourcing character traits at scale has proven productive for both psychological and computational analysis. The Open-Source Psychometrics Project's character ratings, from which the Archetypometrics dataset is derived, have been used to study personality structure in fictional characters (Dahl et al., 2023), the relationship between character traits and narrative role, and the alignment between fictional characters and real-world personality models such as the Big Five. The bipolar trait format (e.g., *playful :: serious*, *goofy :: unfrivolous*) is well-suited to capturing the contrastive nature of character perception.

**Genre and narrative structure.** Genre has been studied computationally through text (Stamatatos, 2009; Biber and Conrad, 2009), through plot structure (Sack, 2011), and through sentiment trajectories (Reagan et al., 2016). Less attention has been paid to genre as a property of character composition rather than linguistic or plot-level features. Closest to our approach is work on screenplay analysis (Gorinski and Lapata, 2015; Kar and Bhatt, 2018), which links character dialogue and network structure to genre, and the study by Skowron et al. (2016) on personality-based genre prediction. We extend this line of work by using a richer, crowd-validated trait space and focusing on the distributional properties of full cast compositions rather than individual protagonist profiles.

**Kolmogorov–Smirnov tests for distributional comparison.** The two-sample KS test has been widely used in computational social science and cultural analytics to compare empirical distributions without parametric assumptions (Clauset et al., 2009). We use it here to characterize differences between comedy and drama character populations both at the archetype level and across individual trait dimensions.

---

## Methods

### Dataset

Character trait data are drawn from the Archetypometrics dataset (Dodds et al.), containing ratings for 2,000 fictional characters across 341 stories. Each character is described by a 464-dimensional vector of bipolar trait scores derived from crowd-sourced ratings on a continuous scale. Trait dimensions take the form of opposing adjective pairs (e.g., *goofy :: unfrivolous*, *chatty :: reserved*), where higher scores correspond to the left (first) pole. Genre labels are assigned at the story level using IMDb genre classifications, matched to story names via a curated mapping file (`story_character_summary.csv`). Stories with a primary genre label of "Comedy" or "Drama" are retained; all others are excluded from this analysis. This yields 73 comedy stories (449 characters, mean 6.2 characters per story) and 55 drama stories (330 characters, mean 6.0 characters per story).

### Archetype Space Construction

Following Dodds et al., we construct a low-dimensional archetype space from the raw trait matrix $\mathbf{A} \in \mathbb{R}^{464 \times 2000}$ via truncated SVD: $\mathbf{A} \approx \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$. The projected archetype coordinates for each character are given by $\mathbf{U}^\top \mathbf{A}$, yielding a representation $\in \mathbb{R}^{464 \times 2000}$, of which we use the first six dimensions. These six dimensions have been labeled by Dodds et al. as bipolar archetype axes: Fool–Hero, Angel–Demon, Traditionalist–Adventurer, Lone Wolf–Diva, Outcast–Sophisticate, and Brute–Geek, corresponding to the dominant axes of variation in the character trait space.

### Genre-Level Comparison

Comedy and drama character populations are defined by collecting all character indices belonging to stories of each genre. For each of the six archetype dimensions and each of the 464 raw trait dimensions, we apply a two-sample Kolmogorov–Smirnov (KS) test to assess whether the comedy and drama character distributions differ significantly. The KS statistic $D_{\text{KS}}$ measures the maximum absolute difference between empirical CDFs and serves as an effect size; we report both $D_{\text{KS}}$ and the corresponding $p$-value.

To compare the spread of comedy and drama characters in archetype space, we compute bootstrapped radii of gyration: we repeatedly sample 500 characters with replacement from each genre's archetype coordinates (6 dimensions), compute the radius of gyration (root mean squared distance from the centroid), and record the distribution over 5,000 bootstrap iterations.

### Trait-Level Analysis

For the 464 raw trait dimensions, we compute: (1) the mean trait score for comedy and drama characters separately; (2) the signed mean difference (comedy − drama); and (3) the KS statistic and $p$-value for each trait. Traits are ranked by absolute mean difference and by KS statistic to identify the most genre-discriminating dimensions. For the top six traits by KS statistic, we plot kernel density estimates (KDEs) of the full trait score distributions for each genre, with mean lines overlaid.

We also produce mirrored histogram visualizations for each archetype dimension, plotting comedy character densities above zero and drama character densities reflected below zero, with median markers, to facilitate visual comparison of distributional shape and location.

---

## Findings

### Archetype Dimensions

Four of six archetype dimensions show statistically significant distributional differences between comedy and drama characters (Table 1). The strongest separation occurs on the **Fool–Hero** axis ($D_{\text{KS}} = 0.170$, $p < 0.0001$): drama characters lean substantially toward the Hero pole (median = 1.61) while comedy characters cluster nearer the Fool pole (median = 0.92). This is consistent with the functional role of comedy protagonists, who frequently derive humor from incompetence, obliviousness, or social failure, while drama protagonists are more often purposeful and capable agents.

The **Traditionalist–Adventurer** axis shows the second-strongest separation ($D_{\text{KS}} = 0.163$, $p < 0.0001$), with comedy characters leaning more toward the Adventurer pole (median = 0.85) and drama characters toward Traditionalist (median = 0.37). Comedy casts tend to feature characters who are unconventional, boundary-crossing, or out-of-place—a structural feature of comedic situations—while drama characters more often occupy conventional social roles. The **Lone Wolf–Diva** axis also shows a significant difference ($D_{\text{KS}} = 0.135$, $p = 0.002$), with comedy characters skewing toward the Diva pole (median = 0.85 vs. 0.56), suggesting greater expressiveness and social prominence in comedy casts. The **Angel–Demon** ($p = 0.364$), **Outcast–Sophisticate** ($p = 0.088$), and **Brute–Geek** ($p = 0.725$) dimensions do not reach conventional significance thresholds, indicating these axes of variation do not cleanly track genre.

**Table 1.** Archetype dimension comparisons, comedy vs. drama.

| Dimension | $D_{\text{KS}}$ | $p$-value | Comedy median | Drama median |
|---|---|---|---|---|
| Fool–Hero | 0.170 | < 0.0001 | 0.92 | 1.61 |
| Traditionalist–Adventurer | 0.163 | < 0.0001 | 0.85 | 0.37 |
| Lone Wolf–Diva | 0.135 | 0.0017 | 0.85 | 0.56 |
| Outcast–Sophisticate | 0.089 | 0.088 | −0.46 | −0.39 |
| Angel–Demon | 0.066 | 0.364 | −0.03 | 0.03 |
| Brute–Geek | 0.049 | 0.725 | −0.20 | −0.20 |

### Trait-Level Differences

Of the 464 bipolar trait dimensions, 304 (65.5%) show statistically significant distributional differences between comedy and drama characters at $p < 0.05$. The most genre-discriminating traits by KS statistic are shown in Table 2.

The trait with the largest effect size is **goofy :: unfrivolous** ($D_{\text{KS}} = 0.355$): comedy characters score substantially more toward the goofy pole. This is followed by **open-book :: secretive** ($D_{\text{KS}} = 0.338$, with drama characters more secretive), **weird :: normal** ($D_{\text{KS}} = 0.316$, comedy characters more eccentric), and **happy :: sad** ($D_{\text{KS}} = 0.295$, comedy characters happier). The trait **dramatic :: comedic** ($D_{\text{KS}} = 0.280$) provides a near-direct genre label in character form: drama characters are rated more dramatic while comedy characters are rated more comedic, confirming that the trait space contains genre-relevant signal.

By signed mean difference, comedy characters are also more **gossiping** (vs. confidential), more **chatty** (vs. reserved), more **clumsy**, more **goof-off** (vs. studious), more **expressive** (vs. stoic), more **awkward** (vs. charming), and more **loud** (vs. quiet). Drama characters, in contrast, score higher on traits associated with gravity and interiority: they are more **secretive**, more **profound** (vs. ironic), more **deep** (vs. epic), more **stoic** (vs. hypochondriac), and more **haunted** (vs. blissful).

These patterns collectively describe comedy characters as socially visible, emotionally exuberant, and behaviorally erratic, while drama characters present as psychologically complex, controlled, and purposeful—a character-level signature of the emotional and narrative demands each genre places on its cast.

**Table 2.** Top 10 traits by KS statistic, comedy vs. drama.

| Trait (left :: right pole) | $D_{\text{KS}}$ | Mean diff. (C−D) | Direction |
|---|---|---|---|
| unfrivolous :: goofy | 0.355 | −0.163 | Drama more unfrivolous |
| open-book :: secretive | 0.338 | +0.140 | Comedy more open-book |
| weird :: normal | 0.316 | +0.138 | Comedy more weird |
| profound :: ironic | 0.306 | −0.097 | Drama more profound |
| deep :: epic | 0.300 | −0.107 | Drama more deep |
| happy :: sad | 0.295 | +0.131 | Comedy more happy |
| stoic :: hypochondriac | 0.293 | −0.110 | Drama more stoic |
| blissful :: haunted | 0.283 | +0.112 | Comedy more blissful |
| jovial :: noble | 0.283 | +0.111 | Comedy more jovial |
| confidential :: gossiping | 0.282 | −0.149 | Drama more confidential |

---

## Discussion

The results establish that genre is systematically encoded in character trait space. Comedy and drama are not merely distinguished by plot structure or linguistic register—they recruit recognizably different kinds of characters, and those differences are measurable, consistent across a large and diverse corpus, and interpretable in terms of long-standing intuitions about how the two genres work.

The finding that comedy characters cluster toward the Fool end of the Fool–Hero axis is one of the most structurally meaningful results. The Fool is defined in the Archetypometrics framework by traits of weakness, incompetence, laziness, and low intelligence—precisely the characteristics that generate comedy through gap between social expectation and actual performance. The bumbling protagonist, the well-meaning idiot, the person who doesn't know what they don't know: these are comedy archetypes because their failure is legible, predictable, and safe. Drama, by contrast, requires characters whose agency drives stakes; incompetence there reads as tragedy rather than farce.

The Traditionalist–Adventurer split is subtler but coherent. Comedy frequently operates by placing characters in situations where conventional rules do not apply—the workplace comedy, the fish-out-of-water narrative, the ensemble of eccentrics navigating a world that doesn't quite fit them. The Adventurer pole captures this resistance to convention. Drama characters, meanwhile, are more often embedded in the weight of social structure: they struggle *within* institutions, families, and norms rather than gleefully against them.

At the trait level, the dominance of social and behavioral traits—goofiness, chattiness, expressiveness, awkwardness—over cognitive or moral traits in distinguishing the genres suggests that comedy is primarily a *register of social performance*. Comedy characters are louder, messier, more visible, and more prone to self-display than their drama counterparts, who operate with greater interiority and restraint. The finding that drama characters are more *secretive* and more *haunted* while comedy characters are more *open-book* and *blissful* points to a deeper asymmetry: drama is the genre of hidden things—hidden motives, hidden wounds, unspoken knowledge—while comedy thrives on exposure and legibility.

Several limitations should be noted. First, genre labels are assigned at the story level and inherited by all characters in a story, which may misclassify characters in tonally mixed works. Second, the IMDb genre taxonomy is itself imprecise: comedies in our corpus range from sitcoms to dark comedies to romantic comedies, and dramas span prestige television to literary adaptations. Third, the trait ratings in the Archetypometrics dataset are crowd-sourced and may reflect cultural biases about character types rather than the characters themselves. Fourth, with 330–449 characters per genre, effect sizes are moderate and some detected differences may reflect corpus composition (e.g., genre-specific prevalence of ensemble casts vs. single protagonists) rather than structural genre properties.

Future work could extend this analysis to subtler genre distinctions (e.g., comedy vs. drama vs. thriller), examine whether genre prediction from character composition generalizes to held-out stories, and investigate how cast composition—not just the traits of individual characters but their combination and contrast—functions as a genre signal. The Fool-heavy comedy cast and the Hero-heavy drama cast are not independent observations; they reflect casting logics in which genre is produced by a *system* of characters rather than any single archetype.

---

*Word counts (approximate): Introduction ~350, Related Works ~450, Methods ~600, Findings ~550, Discussion ~550.*
