"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte; die Zahlen der Hilfetexte."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kxp_constants as C
import kxp_evaluation as ev
import kxp_forms as A
import kxp_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DEFAULT = "Standardfall: ℓ₃-Kugel"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert preset["family"] in A.FAMILIES and preset["p"] in C.P_CHOICES and preset["beta"] in C.BETA_CHOICES
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.STRETCH_MIN <= preset["stretch"] <= C.STRETCH_MAX
        assert C.ANGLE_MIN <= preset["angle"] <= C.ANGLE_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_each_preset_uses_what_its_name_announces():
    pr = C.PRESETS
    assert pr["Die Kugel als Potenzkegel"]["p"] == 2.0 and pr["Die Raute (p = 1)"]["p"] == 1.0 and pr["Fast der Würfel: p = 64"]["p"] == 64.0
    assert pr["Der glatte Würfel: β = 10"]["family"] == "lse" and pr["Der glatte Würfel: β = 10"]["beta"] == 10.0 and pr["Weich: β = 1"]["beta"] == 1.0 and pr[DEFAULT]["p"] == 3.0


def test_the_help_texts_name_numbers_that_the_presets_produce():
    """Bänder statt Punktwerte, wo die Rundung von der CPU abhängt."""
    std = ev.analyse(ev.Settings(**C.PRESETS[DEFAULT]))
    assert std.form.problem.nu == 140 and "ν = 3" in C.PRESET_HELP[DEFAULT] and "a = 1/3" in C.PRESET_HELP[DEFAULT]
    ball = ev.analyse(ev.Settings(**C.PRESETS["Die Kugel als Potenzkegel"]))
    assert ball.form.problem.nu == 140 and "ν = 140 statt 40" in C.PRESET_HELP["Die Kugel als Potenzkegel"]
    assert abs(ball.form.result.value - ball.others["ball"]) < 1e-9
    near = ev.analyse(ev.Settings(**C.PRESETS["Fast der Würfel: p = 64"]))
    assert near.form.result.value - near.cube < 1e-7 and "weniger als 10⁻⁷" in C.PRESET_HELP["Fast der Würfel: p = 64"]
    smooth = ev.analyse(ev.Settings(**C.PRESETS["Der glatte Würfel: β = 10"]))
    assert smooth.form.result.value - smooth.cube < 1e-7 and "unter 10⁻⁷" in C.PRESET_HELP["Der glatte Würfel: β = 10"]
    soft = ev.analyse(ev.Settings(**C.PRESETS["Weich: β = 1"]))
    assert soft.form.result.value - soft.cube == pytest.approx(0.11, abs=0.005) and "um 0,11 größer" in C.PRESET_HELP["Weich: β = 1"]
    import math
    assert math.log(4.0) == pytest.approx(1.39, abs=0.005) and "1,39" in C.PRESET_HELP["Weich: β = 1"]


def test_the_diamond_preset_reaches_the_lp_value():
    a = ev.analyse(ev.Settings(**C.PRESETS["Die Raute (p = 1)"]))
    assert a.form.result.value == pytest.approx(a.others["diamond"], abs=1e-8) and not a.form.original_problems


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["family"] = "lse"
    at.query_params["n"] = "999"            # über der Grenze: wird begrenzt
    at.query_params["stretch"] = "nan"      # keine endliche Zahl: bleibt Standard
    at.query_params["angle"] = "abc"        # keine Zahl: bleibt Standard
    at.query_params["p"] = "5"              # kein erlaubter Wert: bleibt Standard
    at.query_params["beta"] = "30"
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox(key="family_select").value == "lse"
    assert at.sidebar.slider(key="n_slider").value == C.N_MAX
    assert at.sidebar.slider(key="stretch_slider").value == C.DEFAULT_STRETCH
    assert at.sidebar.slider(key="angle_slider").value == C.DEFAULT_ANGLE
    assert at.session_state["p_select"] == C.DEFAULT_P and at.session_state["beta_select"] == 30.0
    bad = AppTest.from_file(APP, default_timeout=180)
    bad.query_params["family"] = "dreieck"
    bad.run()
    assert not bad.exception and bad.sidebar.selectbox(key="family_select").value == C.DEFAULT_FAMILY
