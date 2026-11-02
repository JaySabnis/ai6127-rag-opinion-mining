"""Group C: API sentiment classification, evidence summaries and verification."""

from generation.pipeline import OpinionPipeline, enrich_reply
from generation.sentiment import SentimentClassifier, aggregate_sentiment
from generation.verifier import FaithfulnessChecker

__all__ = ["OpinionPipeline", "enrich_reply", "SentimentClassifier",
           "aggregate_sentiment", "FaithfulnessChecker"]
