"""
mlJudge.py
----------
This module is the "Predictive Regression Model" piece of the project.

CONCEPT OVERVIEW:
A machine learning regression model can only work with numbers -- it has no idea
what a "persuasive argument" is. So our job is two-fold:
  1. FEATURE ENGINEERING: turn raw debate text into a small set of numeric
     signals (extract_NLP_features).
  2. SUPERVISED LEARNING: show a RandomForestRegressor many past examples of
     (features -> human score) so it learns the mapping, then use it to predict
     a score for brand-new, never-before-seen debate text (predict_score).
"""

import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------------------------------
# WHY A HAND-ROLLED SENTIMENT LEXICON INSTEAD OF NLTK/VADER:
# The README lists NLTK/VADER as a BONUS, not a requirement, and it needs an
# extra pip install + a one-time corpus download (nltk.download('vader_lexicon')),
# which can silently fail on locked-down or offline machines during a live demo.
# A small hand-built lexicon has zero extra dependencies and is transparent/
# explainable in the video demo. If you want the bonus points, see the
# UPGRADE PATH note at the bottom of extract_NLP_features().
# --------------------------------------------------------------------------
POSITIVE_WORDS = {
    "innovative", "opportunity", "growth", "beneficial", "empower", "efficient",
    "progress", "solution", "advantage", "improve", "success", "positive",
    "strong", "reliable", "trust", "valuable", "effective", "confident", "gain",
    "breakthrough", "thrive", "boost", "excellent", "promising", "great",
}
NEGATIVE_WORDS = {
    "unemployment", "risk", "danger", "threat", "harmful", "loss", "fail",
    "crisis", "collapse", "damage", "disaster", "flawed", "weak", "dangerous",
    "problem", "worse", "decline", "concern", "fear", "destroy", "negative",
    "bias", "biased", "unfair", "displace", "displacement",
}


class DebateRegressionJudge:
    def __init__(self):
        self.model = None  # Trained sklearn model lives here once train_model() runs
        # Keep the exact column order used at training time so predict_score()
        # always feeds the model features in the same order it learned them in.
        self.feature_columns = ["word_count", "complexity_score", "sentiment_score"]

    def extract_NLP_features(self, text: str) -> dict:
        """
        Converts raw debate text into the 3 numeric features the model expects.
        """
        # --- word_count ---
        words = text.split()
        word_count = len(words)

        # --- complexity_score (0-10 scale) ---
        # Blends two classic readability/complexity signals:
        #   (a) average sentence length -- longer sentences are usually more complex
        #   (b) vocabulary richness (unique words / total words) -- more varied
        #       word choice signals a more sophisticated argument
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        avg_sentence_len = (word_count / len(sentences)) if sentences else word_count

        unique_words = set(w.strip(".,!?\"'").lower() for w in words)
        vocab_richness = (len(unique_words) / word_count) if word_count > 0 else 0

        # Normalize both signals onto a rough 0-10 scale, then average them.
        # (Sentence length of ~25 words and vocab richness of ~0.8 are treated
        # as "maximally complex" ceilings for this project's purposes.)
        sentence_len_component = min(avg_sentence_len / 25.0, 1.0) * 10
        vocab_component = min(vocab_richness / 0.8, 1.0) * 10
        complexity_score = round((sentence_len_component + vocab_component) / 2, 2)

        # --- sentiment_score (-1 to 1 scale) ---
        # Simple bag-of-words polarity: count positive vs negative lexicon hits,
        # normalized by text length so short and long texts are comparable.
        lowered = [w.strip(".,!?\"'").lower() for w in words]
        pos_hits = sum(1 for w in lowered if w in POSITIVE_WORDS)
        neg_hits = sum(1 for w in lowered if w in NEGATIVE_WORDS)
        if pos_hits + neg_hits == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = round((pos_hits - neg_hits) / (pos_hits + neg_hits), 3)

        # UPGRADE PATH (bonus points): swap the block above for VADER --
        #   from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        #   sentiment_score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
        # Requires: pip install vaderSentiment

        return {
            "word_count": word_count,
            "complexity_score": complexity_score,
            "sentiment_score": sentiment_score,
        }

    def train_model(self, dataset_path: str) -> dict:
        """
        Trains the RandomForestRegressor on historical (already-labeled) debate data.
        """
        print(f"Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)

        # X = the input features the model learns FROM.
        # y = the target value the model learns to PREDICT.
        X = df[self.feature_columns]
        y = df["human_persuasiveness_score"]

        # Hold back 20% of the data purely for honest evaluation -- the model
        # never sees this slice during .fit(), so testing on it tells us how
        # well the model generalizes to debates it hasn't memorized.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # n_estimators=100: number of decision trees averaged together (more
        # trees = more stable predictions, at the cost of training time).
        # random_state=42: makes results reproducible for grading/demo purposes.
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # MSE: average squared difference between predicted and true scores
        # (lower is better; 0 = perfect predictions).
        mse = mean_squared_error(y_test, predictions)
        # R^2: fraction of the variance in human scores our model explains
        # (1.0 = perfect, 0.0 = no better than guessing the average every time).
        r2 = r2_score(y_test, predictions)

        self.model = model
        print("Model Trained!")

        return {"mse": round(float(mse), 4), "r2_score": round(float(r2), 4)}

    def predict_score(self, text: str) -> float:
        """
        Called LIVE during the debate to score one agent's argument turn.
        """
        if self.model is None:
            raise Exception("Model is not trained yet! Call train_model() first.")

        features = self.extract_NLP_features(text)

        # Build a single-row DataFrame with columns in the SAME order used
        # during training -- sklearn matches features by position, not by name,
        # so column order mismatches silently produce garbage predictions.
        feature_row = pd.DataFrame([[features[col] for col in self.feature_columns]],
                                    columns=self.feature_columns)

        raw_prediction = self.model.predict(feature_row)[0]

        # Clip to the 1-10 rubric scale the UI expects, in case the model
        # extrapolates slightly outside the training range.
        final_score = round(float(np.clip(raw_prediction, 1, 10)), 2)
        return final_score