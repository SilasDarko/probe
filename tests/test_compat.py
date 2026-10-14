"""Tests for environment compatibility checks (probe.compat)."""

from probe.compat import check_compatibility

from synthetic import make_result


def test_identical_environments_are_compatible():
    c = check_compatibility(make_result(), make_result())
    assert (c.fatal, c.environment, c.warnings) == ([], [], [])
    assert not c.blocked(allow_environment_mismatch=False)


def test_different_commit_is_not_a_mismatch():
    c = check_compatibility(make_result(commit="a" * 40), make_result(commit="b" * 40))
    assert (c.fatal, c.environment, c.warnings) == ([], [], [])


def test_workload_mismatch_is_fatal_and_not_overridable():
    c = check_compatibility(make_result(workload="vector_add"), make_result(workload="reduction"))
    assert any(line.startswith("workload:") for line in c.fatal)
    assert c.blocked(allow_environment_mismatch=False)
    assert c.blocked(allow_environment_mismatch=True)


def test_parameter_mismatch_is_fatal_and_not_overridable():
    c = check_compatibility(make_result(n=1000), make_result(n=2000))
    assert c.fatal == ["parameter n: baseline 1000, current 2000"]
    assert c.blocked(allow_environment_mismatch=True)


def test_missing_parameter_is_a_mismatch():
    current = make_result()
    del current["parameters"]["dtype"]
    assert c_fatal(make_result(), current) == ["parameter dtype: baseline 'float32', current None"]


def c_fatal(base, cur):
    return check_compatibility(base, cur).fatal


def test_gpu_mismatch_is_overridable():
    c = check_compatibility(make_result(gpu="GPU A"), make_result(gpu="GPU B"))
    assert c.fatal == []
    assert c.environment == ["GPU name: baseline 'GPU A', current 'GPU B'"]
    assert c.blocked(allow_environment_mismatch=False)
    assert not c.blocked(allow_environment_mismatch=True)


def test_build_type_mismatch_is_overridable():
    c = check_compatibility(make_result(build_type="Release"), make_result(build_type="Debug"))
    assert c.environment == ["build type: baseline 'Release', current 'Debug'"]
    assert c.blocked(allow_environment_mismatch=False)


def test_driver_difference_is_only_a_warning():
    c = check_compatibility(make_result(driver="12.2"), make_result(driver="12.4"))
    assert c.fatal == [] and c.environment == []
    assert c.warnings == ["cuda driver version: baseline '12.2', current '12.4'"]
    assert not c.blocked(allow_environment_mismatch=False)


def test_iteration_count_difference_is_only_a_warning():
    c = check_compatibility(make_result(iterations=20), make_result(iterations=30))
    assert c.warnings == ["iterations: baseline 20, current 30"]
    assert not c.blocked(allow_environment_mismatch=False)
