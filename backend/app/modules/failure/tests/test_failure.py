from backend.app.modules.failure.schemas import FailureThresholds, FailureType
from backend.app.modules.failure.service import _classify


def test_high_shift_low_drop_is_false_alarm():
    failure_type, _ = _classify(92, 0.02, FailureThresholds())
    assert failure_type == FailureType.FALSE_ALARM


def test_low_shift_high_drop_is_miss():
    failure_type, _ = _classify(35, 0.28, FailureThresholds())
    assert failure_type == FailureType.MISS


def test_consistent_values_are_ok():
    failure_type, _ = _classify(50, 0.10, FailureThresholds())
    assert failure_type == FailureType.OK
