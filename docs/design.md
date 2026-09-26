# Design notes

## Problem

A scam-detection system that only says "scam / not scam" is not enough for
the teams downstream. Product, policy and user-education teams need to know
*which kind* of scam is growing (parcel-delivery lures vs. bank impersonation
vs. "hi mum, new phone") to decide what to block, what to warn users about,
and which brands to notify. This repo builds that second-stage classifier:
given a message already flagged as smishing, assign a scam type.

## Data

IMC'25 "Fishing for Smishing" dataset (Agarwal et al., CC BY 4.0): ~34k
user-reported smishing messages in 50+ languages, labeled with 8 scam types.
PII is already masked with placeholders (`<URL>`, `<PHONE_NUMBER>`,
`<NAMED_ENTITY>`). Two properties drive every design decision below:

1. **Heavy class imbalance.** `banking` is ~45% of messages; `wrong_number`
   and `family_impersonation` are each under 1%.
2. **Campaign duplication.** Scammers send the same template thousands of
   times with small edits (a name, a link, an amount). A quarter of the rows
   are exact duplicates after normalization, and many more are near-duplicates.

## Decisions and alternatives considered

### 1. Split by template, not by row
*Decision:* cluster near-duplicates with MinHash LSH (character 5-gram
shingles, Jaccard >= 0.8), then use a stratified *group* split so every
template is entirely in train or entirely in test.

*Why:* with a row-level split, copies of the same campaign sit on both sides
and the metric rewards memorization. In production the model mostly meets
*new* campaigns, so the grouped split is the honest offline proxy. The
results report both numbers so the size of the leak is visible.

*Rejected:* dropping all duplicates before splitting. It fixes leakage but
throws away information about how much volume each campaign carries, which
matters for traffic-weighted metrics.

### 2. Cap copies per template in training
*Decision:* keep at most 5 messages per template in the training set.

*Why:* one mass campaign can contribute hundreds of identical rows, which
acts as an unintended sample weight and pulls the decision boundary toward
that campaign. Capping keeps variety without letting volume dominate.

### 3. Class-balanced loss instead of oversampling
*Decision:* `class_weight="balanced"` in logistic regression.

*Rejected:* random oversampling duplicates minority rows (which the template
cap is trying to undo), and synthetic text augmentation needs an LLM in the
loop, which is out of scope for a fully reproducible, offline baseline.

### 4. Character n-grams for 50+ languages
*Decision:* word (1–2) + character (2–5, word-boundary) TF-IDF.

*Why:* no per-language tokenizer is needed, the model trains on CPU in
minutes, and character n-grams are robust to the deliberate misspellings
scammers use to dodge filters. A multilingual transformer (e.g. XLM-R) is
the obvious next step; this baseline sets the bar it has to beat.

### 5. Hierarchical model with *learned* groups
*Decision:* fit a flat model with cross-validation, cluster classes by
symmetric confusion (average-linkage), route to a group, then resolve within
the group with a specialist. Probabilities combine as
P(class) = P(group) · P(class | group).

*Why:* in fine-grained abuse taxonomies a few classes (here, the `others`
catch-all) absorb errors from everything else. A router plus specialists is a
common fix, and learning the groups from the confusion matrix avoids
hand-picking them. Whether it beats the flat model is an empirical question;
see the results for the answer on this dataset.

### 6. Evaluate like an operations team would
- **Macro-F1** as the headline, because small classes (family impersonation)
  are often the highest-harm ones.
- **Cluster bootstrap CIs**: resample templates, not rows, since messages in
  a template are not independent.
- **One-message-per-template score**: how well the model generalizes across
  campaigns rather than across copies.
- **Per-language slices**, to catch a model that only works in English.
- **Review routing curve**: auto-label only confident predictions and send
  the rest to human reviewers; report coverage vs. quality at each threshold.

## Limitations
- Labels come from the dataset authors; I did not re-audit them, and
  `others` is a heterogeneous catch-all that caps achievable accuracy.
- User-reported data over-represents scams people notice and bother to
  report.
- No temporal split: the dataset spans a limited window, so drift across
  campaign seasons is not measured.
