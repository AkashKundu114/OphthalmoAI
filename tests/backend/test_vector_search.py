"""
Tests for backend/vector_search.py (Vector Search & Similar Case Retrieval).
"""

import numpy as np
import pytest

from backend.vector_search import (
    ClinicalCaseVectorIndex,
    HISTORICAL_CLINICAL_ARCHIVE,
    vector_index,
)


def test_vector_index_initialization():
    idx = ClinicalCaseVectorIndex(dim=512)
    assert len(idx.cases) == len(HISTORICAL_CLINICAL_ARCHIVE)
    assert idx.vectors.shape == (len(HISTORICAL_CLINICAL_ARCHIVE), 512)


def test_query_similar_cases_exact_match():
    # Query with exact embedding of Diabetic Retinopathy case
    target_vec = vector_index.cases[0]["embedding"]
    results = vector_index.query_similar_cases(target_vec, top_k=2)

    assert len(results) >= 1
    top_match = results[0]
    assert top_match["case_id"] == "REF-DR-104"
    assert top_match["diagnosis"] == "Diabetic Retinopathy"
    assert top_match["similarity_score"] >= 99.0
    assert "treatment_protocol" in top_match
    assert "outcome_12mo" in top_match


def test_query_from_probabilities():
    probs = {"Glaucoma": 0.92, "Normal": 0.08}
    query_vec = vector_index.extract_embedding_from_probabilities(probs)
    assert query_vec.shape == (512,)

    results = vector_index.query_similar_cases(query_vec, top_k=3)
    assert len(results) >= 1
    assert results[0]["diagnosis"] == "Glaucoma"
    assert results[0]["similarity_score"] > 80.0
