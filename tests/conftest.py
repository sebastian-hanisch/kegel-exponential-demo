"""Gemeinsame Fixtures: die Mini-Programme, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kxp_barrier as B  # noqa: E402


@pytest.fixture
def minis():
    """Zwei Mini-Programme mit Handoptimum.

    exp: min z mit (x, 1, z) im Exponentialkegel (also e^x <= z) und x >= 1: Optimum e bei x = 1.
    pow: min -w mit (2, 8, w) im Potenzkegel a = 1/2 (|w| <= sqrt(2 * 8) = 4): Optimum -4."""
    exp = B.Program(np.array([0.0, 1.0]), np.array([[-1.0, 0.0], [0.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]), np.array([0.0, 1.0, 0.0, -1.0]), (("e", None, 1), ("l", None, 1)), np.array([2.0, 20.0]))
    pw = B.Program(np.array([-1.0]), np.array([[0.0], [0.0], [-1.0]]), np.array([2.0, 8.0, 0.0]), (("p", 0.5, 1),), np.array([0.0]))
    return {"exp": (exp, math.e), "pow": (pw, -4.0)}


@pytest.fixture
def antipodal():
    """Zwei Punkte (1, 2) und (-1, -2): der Mittelpunkt ist 0 (Symmetrie und Konvexität), der Radius das Maß von (1, 2)."""
    return np.array([[1.0, 2.0], [-1.0, -2.0]])


@pytest.fixture
def hand():
    """Vier Punkte (1,1), (-1,-1), (0,5; -0,5), (-0,5; 0,5), wie in den früheren Stücken der Reihe (der Kernprüfung gemeinsam)."""
    return np.array([[1.0, 1.0], [-1.0, -1.0], [0.5, -0.5], [-0.5, 0.5]])


@pytest.fixture
def tiny_cones():
    """Drei Mini-Kegelprogramme des Kerns mit bekanntem Optimum, je ein Kegel (test_kern.py, wortgleich in der Reihe):
    Orthant: min x mit x >= 1 -> 1;  Lorentz: min t mit (t; 3, 4) im Kegel -> 5;  PSD: min x mit [[x, 1],[1, x]] >= 0 -> 1."""
    import kegel_kern as K

    sq2 = K.SQ2
    lp = K.ConeProblem(np.array([1.0]), np.array([[-1.0]]), np.array([-1.0]), (("l", 1),), np.array([2.0]))
    soc = K.ConeProblem(np.array([1.0]), np.array([[-1.0], [0.0], [0.0]]), np.array([0.0, 3.0, 4.0]), (("q", 3),), np.array([6.0]))
    sdp = K.ConeProblem(np.array([1.0]), np.array([[-1.0], [0.0], [-1.0]]), np.array([0.0, sq2, 0.0]), (("s", 2),), np.array([2.0]))
    return {"lp": (lp, 1.0), "soc": (soc, 5.0), "sdp": (sdp, 1.0)}
