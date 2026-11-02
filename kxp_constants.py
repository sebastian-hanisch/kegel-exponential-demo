"""Konstanten: Regler-Grenzen, feste Stichproben, Farben und Presets."""

import kxp_forms as A

DEFAULT_FAMILY = "lp"
N_MIN, N_MAX, DEFAULT_N = 8, 40, 20
STRETCH_MIN, STRETCH_MAX, DEFAULT_STRETCH = 1.2, 6.0, 4.0
ANGLE_MIN, ANGLE_MAX, DEFAULT_ANGLE = 0, 90, 45
SEED_MAX, DEFAULT_SEED = 999999, 35
P_CHOICES = (1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)       # Exponent der ℓp-Kugel (Potenzkegel a = 1/p)
DEFAULT_P = 3.0
BETA_CHOICES = (0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0)                  # Schärfe der log-sum-exp-Kugel (Exponentialkegel)
DEFAULT_BETA = 10.0
SWEEP_SEEDS = tuple(range(100000, 100003))                                  # feste Wolken für die Sweeps (drei): nie ein einzelner Seed als Befund
TOL = 1e-9                                                                  # Genauigkeit der Barrierenmethode, relativ zum Zielwert des Starts
N_SWEEP = (4, 10, 20, 40)                                                   # Punktzahlen in Experiment 5
MUS = (2.0, 8.0, 32.0, 1000.0)                                              # Schrittfaktoren in Experiment 5
P_GRID = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)      # Experiment 2
BETA_GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1000.0)            # Experiment 3
PATH_TS = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7)                  # Pfadpunkte in Experiment 1
SYMMETRY_POINTS = 400                                                       # Randpunkte je Kegel für die Automorphismen-Probe
DUAL_SAMPLES = 400
PAIR_SAMPLES = 20000
SCALE_FACTORS = (1e-6, 1.0, 1e3, 1e6)                                       # Datenmaßstab

COLORS = {"lp": "#2F6B65", "lse": "#4c78a8", "soc": "#888888", "bound": "#c0504d", "cube": "#b08a2e", "diamond": "#8c6bb1", "ball": "#888888"}
COLOR_OK = "#2F6B65"
COLOR_WARN = "#b08a2e"
COLOR_BAD = "#b34a4a"

PRESETS = {
    "Standardfall: ℓ₃-Kugel": {"family": "lp", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "p": 3.0, "beta": DEFAULT_BETA},
    "Die Kugel als Potenzkegel": {"family": "lp", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "p": 2.0, "beta": DEFAULT_BETA},
    "Die Raute (p = 1)": {"family": "lp", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "p": 1.0, "beta": DEFAULT_BETA},
    "Fast der Würfel: p = 64": {"family": "lp", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "p": 64.0, "beta": DEFAULT_BETA},
    "Der glatte Würfel: β = 10": {"family": "lse", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "p": DEFAULT_P, "beta": 10.0},
    "Weich: β = 1": {"family": "lse", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "p": DEFAULT_P, "beta": 1.0},
}
PRESET_HELP = {
    "Standardfall: ℓ₃-Kugel": "20 Punkte, in die Länge gezogen und um 45 Grad gedreht; die kleinste ℓ₃-Kugel ist eine Rechnung mit Potenzkegeln (a = 1/3), je Koordinate ein Kegel mit ν = 3.",
    "Die Kugel als Potenzkegel": "p = 2: derselbe Radius wie die Lorentz-Kugel aus Stück 5, aber mit Potenzkegeln a = 1/2 (ν = 140 statt 40): ν gehört zur Formulierung.",
    "Die Raute (p = 1)": "p = 1: die Raute aus Stück 2 als Potenzkegel mit a = 1 (die Barriere ist in der zweiten Koordinate entartet, das Programm bleibt lösbar).",
    "Fast der Würfel: p = 64": "p = 64: der Radius liegt weniger als 10⁻⁷ über dem Würfelradius (der Würfel ist die Grenze p → ∞); das Programm bleibt gut lösbar.",
    "Der glatte Würfel: β = 10": "Die log-sum-exp-Kugel mit β = 10 liegt schon unter 10⁻⁷ über dem Würfelradius; die Exponentialkegel tragen den glatten Würfel.",
    "Weich: β = 1": "β = 1: die Kugel ist deutlich weicher als der Würfel (Radius um 0,11 größer); die Schranke log(2d)/β = 1,39 ist hier viel zu grob.",
}
assert set(PRESETS) == set(PRESET_HELP) and all(p["family"] in A.FAMILIES and p["p"] in P_CHOICES and p["beta"] in BETA_CHOICES for p in PRESETS.values())
