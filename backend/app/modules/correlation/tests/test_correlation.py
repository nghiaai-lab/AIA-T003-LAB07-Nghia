import pytest

from backend.app.modules.correlation.schemas import CorrelationInputRecord
from backend.app.modules.correlation.service import _compute_spearman


def test_perfect_positive_correlation():
    records = [
        CorrelationInputRecord(domain_or_slice="a", shift_score=10, performance_drop=0.1),
        CorrelationInputRecord(domain_or_slice="b", shift_score=20, performance_drop=0.2),
        CorrelationInputRecord(domain_or_slice="c", shift_score=30, performance_drop=0.3),
    ]

    rho, p_value = _compute_spearman(records)

    assert rho == pytest.approx(1.0)


def test_no_correlation_between_constant_and_varying_values():
    records = [
        CorrelationInputRecord(domain_or_slice="a", shift_score=10, performance_drop=0.5),
        CorrelationInputRecord(domain_or_slice="b", shift_score=10, performance_drop=0.1),
        CorrelationInputRecord(domain_or_slice="c", shift_score=10, performance_drop=0.9),
    ]

    rho, _ = _compute_spearman(records)

    assert rho != rho or rho == pytest.approx(0.0)  # NaN (scipy) or 0 when one series is constant
