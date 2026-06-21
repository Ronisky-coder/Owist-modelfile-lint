from owist_modelfile_lint.params import PARAMETER_SPECS, closest_param_name


def test_known_params_present():
    for name in ["temperature", "top_p", "top_k", "num_ctx", "seed", "stop", "min_p"]:
        assert name in PARAMETER_SPECS


def test_closest_param_name_catches_typo():
    assert closest_param_name("temprature") == "temperature"
    assert closest_param_name("tempurature") == "temperature"
    assert closest_param_name("top_kk") == "top_k"


def test_closest_param_name_no_match_for_nonsense():
    assert closest_param_name("xyz123completely_unrelated") is None


def test_stop_is_repeatable():
    assert PARAMETER_SPECS["stop"].allow_repeat is True


def test_temperature_not_repeatable():
    assert PARAMETER_SPECS["temperature"].allow_repeat is False
