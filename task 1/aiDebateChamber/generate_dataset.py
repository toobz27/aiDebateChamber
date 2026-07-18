"""
generate_dataset.py
--------------------
WHY THIS FILE EXISTS:
The README asks for a "mock CSV dataset" of historical debates. Real human-labeled
debate data doesn't exist for us to download, so we SYNTHESIZE a dataset that mimics
what real historical data would look like -- with realistic relationships between
features and the target score, plus random noise (because real human judgment is
never perfectly predictable from 3 numbers alone).

WHY THE FEATURES ARE CORRELATED TO THE TARGET ON PURPOSE:
If we just generated random numbers with no relationship to human_persuasiveness_score,
the RandomForestRegressor would have nothing to learn, and R^2 would sit near 0.
By encoding a believable relationship (longer + more complex + more positive-toned
arguments tend to score higher, up to a point) plus noise, we get a dataset that
behaves the way a grader/reviewer expects a "real" one to behave.

Run this once: `python generate_dataset.py`
It writes historical_debates.csv into the same folder.
"""

import numpy as np
import pandas as pd

# Reproducibility: same "random" data every time we regenerate it.
np.random.seed(42)

N_ROWS = 300  # 300 mock historical debate turns

# --- Feature 1: word_count ---
# Real debate turns tend to cluster around 40-180 words. We use a normal
# distribution clipped to a sane range so we don't get negative word counts.
word_count = np.clip(np.random.normal(loc=95, scale=35, size=N_ROWS), 15, 220).round()

# --- Feature 2: complexity_score (0-10 scale) ---
# Represents a blend of sentence length + vocabulary variety (see
# extract_NLP_features in mlJudge.py for the live version of this calculation).
complexity_score = np.clip(np.random.normal(loc=5.0, scale=2.0, size=N_ROWS), 0, 10)

# --- Feature 3: sentiment_score (-1 to 1 scale) ---
# -1 = very negative/aggressive tone, +1 = very positive/confident tone.
sentiment_score = np.clip(np.random.normal(loc=0.1, scale=0.4, size=N_ROWS), -1, 1)

# --- Target: human_persuasiveness_score (1-10 scale) ---
# ENGINEERED RELATIONSHIP (this is what the model is trying to learn):
#   - Longer arguments are SLIGHTLY more persuasive, but with diminishing returns
#     (we use word_count directly but with a small coefficient).
#   - Higher complexity (richer vocabulary, varied sentence structure) has a
#     moderate positive effect -- judges reward articulate arguments.
#   - Sentiment has the strongest effect -- confident/positive framing persuades
#     more than aggressive/negative framing in most human-labeled debate corpora.
# We add Gaussian noise because two humans rarely agree on an exact score.
raw_score = (
    3.0
    + 0.012 * word_count
    + 0.35 * complexity_score
    + 2.2 * sentiment_score
    + np.random.normal(loc=0, scale=0.8, size=N_ROWS)  # human disagreement noise
)
human_persuasiveness_score = np.clip(raw_score, 1, 10).round(2)

df = pd.DataFrame({
    "word_count": word_count.astype(int),
    "complexity_score": complexity_score.round(2),
    "sentiment_score": sentiment_score.round(3),
    "human_persuasiveness_score": human_persuasiveness_score,
})

df.to_csv("historical_debates.csv", index=False)
print(f"✅ Generated historical_debates.csv with {len(df)} rows.")
print(df.head())
