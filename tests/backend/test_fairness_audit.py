"""
Tests for Demographic Fairness, Bias Auditing & Slice Disparity Analysis.
"""

import pytest
from backend.fairness_audit import DemographicFairnessAuditor, get_fairness_audit_report


def test_fairness_auditor_evaluation():
    report = DemographicFairnessAuditor.evaluate_synthetic_cohorts(sample_size=600)
    assert report is not None
    assert report["sample_size"] == 600
    assert "overall_test_accuracy" in report
    assert "worst_group_accuracy" in report
    assert "equalized_odds_difference" in report
    assert "disparate_impact_ratio" in report
    assert report["fairness_certified"] is True

    breakdowns = report["cohort_breakdowns"]
    assert "age_cohorts" in breakdowns
    assert "optical_quality_slices" in breakdowns
    assert "comorbidity_slices" in breakdowns

    # Verify age slices
    young = breakdowns["age_cohorts"]["Young (<45)"]
    assert young["tpr"] > 0.8
    assert young["count"] > 0


def test_get_fairness_audit_report_cached_and_refreshed():
    r1 = get_fairness_audit_report()
    r2 = get_fairness_audit_report(force_refresh=False)
    assert r1["timestamp"] == r2["timestamp"]

    r3 = get_fairness_audit_report(force_refresh=True)
    assert r3["fairness_certified"] is True
    assert "compliance_standards" in r3
    assert r3["compliance_standards"]["four_fifths_rule"].startswith("PASSED")
