"""Die Kegel (kxp_cones): Barrieren von Hand und als Eigenschaft, Dualkegel, Automorphismen."""

import math

import numpy as np
import pytest

import kxp_cones as C


def _exp_points(rng, n):
    y = np.exp(rng.normal(0, 0.5, n))
    x = rng.normal(0, 0.5, n)
    z = y * np.exp(x / y) * np.exp(np.abs(rng.normal(size=n)) + 0.05)
    return np.stack([x, y, z], axis=1)


def _pow_points(rng, n, a):
    u, v = np.exp(rng.normal(0, 0.5, n)), np.exp(rng.normal(0, 0.5, n))
    w = u ** a * v ** (1 - a) * rng.uniform(-0.9, 0.9, n)
    return np.stack([u, v, w], axis=1)


def _fn(kind, a):
    return (lambda M: C.exp_grad_hess(M)) if kind == "e" else (lambda M: C.pow_grad_hess(M, a))


def test_exponential_barrier_by_hand_at_zero_one_e_squared():
    """s = (0, 1, e²): ψ = log z = 2, φ = −log 2 − 2, ∇φ = (½, −3/2, −3 e⁻²/2)."""
    val, g, H = C.exp_grad_hess(np.array([[0.0, 1.0, math.e ** 2]]))
    assert val == pytest.approx(-math.log(2.0) - 2.0)
    assert g[0] == pytest.approx([0.5, -1.5, -1.5 * math.e ** -2])
    assert H[0] == pytest.approx(H[0].T) and np.linalg.eigvalsh(H[0]).min() > 0


def test_power_barrier_by_hand_at_two_eight_zero():
    """a = ½, s = (2, 8, 0): Argument u v = 16, φ = −log 16 − ½ log 2 − ½ log 8, ∇φ = −(8, 2, 0)/16 − (¼, 1/16, 0)."""
    val, g, H = C.pow_grad_hess(np.array([[2.0, 8.0, 0.0]]), 0.5)
    assert val == pytest.approx(-math.log(16.0) - 0.5 * math.log(2.0) - 0.5 * math.log(8.0))
    assert g[0] == pytest.approx([-8.0 / 16.0 - 0.25, -2.0 / 16.0 - 1.0 / 16.0, 0.0])


@pytest.mark.parametrize("kind,a", [("e", None), ("p", 0.2), ("p", 0.5), ("p", 0.75)])
def test_barrier_identities_of_a_logarithmically_homogeneous_barrier(kind, a):
    """⟨∇φ, s⟩ = −ν = −3, ∇²φ s = −∇φ, ⟨∇φ, (∇²φ)⁻¹ ∇φ⟩ = 3, φ(λ s) = φ(s) − 3 log λ."""
    rng = np.random.default_rng(1)
    S = _exp_points(rng, 200) if kind == "e" else _pow_points(rng, 200, a)
    val, g, H = _fn(kind, a)(S)
    assert np.allclose(np.einsum("ni,ni->n", g, S), -3.0, atol=1e-9)
    assert np.allclose(np.einsum("nij,nj->ni", H, S), -g, atol=1e-8 * (1 + np.abs(g).max()))
    quad = np.einsum("ni,ni->n", g, np.linalg.solve(H, g[:, :, None])[:, :, 0])
    assert np.allclose(quad, 3.0, atol=1e-8)
    val2, _, _ = _fn(kind, a)(2.5 * S)
    assert val2 == pytest.approx(val - 3.0 * len(S) * math.log(2.5), abs=1e-8)


@pytest.mark.parametrize("kind,a", [("e", None), ("p", 0.3), ("p", 0.5)])
def test_gradient_and_hessian_against_finite_differences_at_moderate_points(kind, a):
    rng = np.random.default_rng(2)
    S = _exp_points(rng, 30) if kind == "e" else _pow_points(rng, 30, a)
    fn = _fn(kind, a)
    for s in S:
        _, g, H = fn(s[None, :])
        for h in (1e-4, 1e-5):
            fd_g = np.array([(fn((s + h * e)[None, :])[0] - fn((s - h * e)[None, :])[0]) / (2 * h) for e in np.eye(3)])
            fd_H = np.array([(fn((s + h * e)[None, :])[1][0] - fn((s - h * e)[None, :])[1][0]) / (2 * h) for e in np.eye(3)])
            assert np.allclose(g[0], fd_g, rtol=1e-5, atol=1e-6 * (1 + np.abs(g).max())), (s, h)
            assert np.allclose(H[0], fd_H, rtol=1e-4, atol=1e-5 * (1 + np.abs(H).max())), (s, h)


def test_the_hessian_is_symmetric_positive_definite_even_close_to_the_boundary():
    rng = np.random.default_rng(3)
    S = _exp_points(rng, 100)
    S[:, 2] = S[:, 1] * np.exp(S[:, 0] / S[:, 1]) * (1.0 + 1e-6)                  # fast auf dem Rand
    _, _, H = C.exp_grad_hess(S)
    assert np.allclose(H, np.swapaxes(H, 1, 2), rtol=1e-9) and np.linalg.eigvalsh(H).min() > 0
    P = _pow_points(rng, 100, 0.3)
    P[:, 2] = np.sign(P[:, 2]) * P[:, 0] ** 0.3 * P[:, 1] ** 0.7 * (1.0 - 1e-6)
    _, _, H = C.pow_grad_hess(P, 0.3)
    assert np.linalg.eigvalsh(H).min() > 0


def test_points_outside_or_on_the_boundary_give_infinity_and_no_derivatives():
    for S in (np.array([[0.0, 1.0, 1.0]]), np.array([[0.0, -1.0, 5.0]]), np.array([[3.0, 1.0, 5.0]]), np.array([[0.0, 1.0, -2.0]])):   # Rand, y < 0, außerhalb, z < 0
        val, g, H = C.exp_grad_hess(S)
        assert val == math.inf and g is None and H is None and not C.exp_inside(S).any()
    for S in (np.array([[1.0, 1.0, 1.0]]), np.array([[1.0, 1.0, 1.5]]), np.array([[-1.0, 1.0, 0.0]])):
        assert C.pow_grad_hess(S, 0.5)[0] == math.inf and not C.pow_inside(S, 0.5).any()
    assert C.exp_inside(np.array([[0.0, 1.0, 1.5]])).all() and C.pow_inside(np.array([[1.0, 1.0, 0.5]]), 0.5).all()


def test_the_power_cone_with_one_half_is_the_rotated_lorentz_cone():
    rng = np.random.default_rng(4)
    S = rng.normal(size=(2000, 3)) * 2.0
    S[:, :2] = np.abs(S[:, :2])
    assert np.array_equal(C.pow_inside(S, 0.5), (S[:, 0] * S[:, 1] > S[:, 2] ** 2) & (S[:, 0] > 0) & (S[:, 1] > 0))


def test_the_dual_exponential_cone_by_hand_and_the_witness():
    assert C.exp_member((1.0, 1.0, 1.5 * math.e)) and not C.dual_exp_member((1.0, 1.0, 1.5 * math.e))              # in K, nicht in K*: K ≠ K*
    assert C.dual_exp_member((-1.0, 1.0, 5.0)) and not C.dual_exp_member((-1.0, 1.0, 0.1))
    assert C.dual_exp_member((0.0, 1.0, 1.0)) and not C.dual_exp_member((0.0, -1.0, 1.0))                            # Abschluss mit u = 0
    # Rand: −u e^{v/u} = e w mit u = −1, v = −1 gibt e = e w, also w = 1
    assert C.dual_exp_member((-1.0, -1.0, 1.0)) and not C.dual_exp_member((-1.0, -1.0, 0.99))


def test_minus_the_gradient_lies_in_the_interior_of_the_dual_cone_and_weak_duality_holds():
    rng = np.random.default_rng(5)
    S = _exp_points(rng, 500)
    _, g, _ = C.exp_grad_hess(S)
    assert C.dual_exp_inside(-g).all()
    y = np.exp(rng.normal(size=20000))
    x = rng.normal(size=20000)
    z = y * np.exp(x / y) * np.exp(np.abs(rng.normal(size=20000)))
    u = -np.exp(rng.normal(size=20000))
    v = rng.normal(size=20000)
    w = (-u * np.exp(v / u) / math.e) * np.exp(np.abs(rng.normal(size=20000)))
    assert np.all(x * u + y * v + z * w >= -1e-12)


@pytest.mark.parametrize("kind,a,expected", [("l", None, 3), ("q", None, 4), ("p", 0.5, 4), ("p", 0.3, 2), ("p", 0.75, 2), ("e", None, 2)])
def test_dimension_of_the_automorphism_group(kind, a, expected):
    dim, sv = C.automorphism_dimension(kind, a, 400, seed=0)
    assert dim == expected and len(sv) == 9
    assert sv[9 - expected - 1] > 1e6 * max(sv[9 - expected], 1e-300)                    # klare Lücke zwischen den kleinen und den großen Singulärwerten
    assert C.automorphism_dimension(kind, a, 300, seed=7)[0] == expected                  # unabhängig von der Stichprobe


def test_homogeneous_cones_are_exactly_those_with_dimension_at_least_three():
    dims = [C.automorphism_dimension(k, a)[0] for k, a in (("l", None), ("q", None), ("p", 0.5), ("p", 0.3), ("e", None))]
    assert [d >= 3 for d in dims] == [True, True, True, False, False]


def test_the_scalings_that_exist_are_in_the_group():
    """Die zwei Erzeuger des Exponentialkegels: (x, y, z) → (x + c y, y, e^c z) und die Streckung erhalten y e^{x/y} ≤ z."""
    rng = np.random.default_rng(6)
    S = _exp_points(rng, 200)
    c, lam = 0.7, 3.0
    T1 = np.stack([S[:, 0] + c * S[:, 1], S[:, 1], math.exp(c) * S[:, 2]], axis=1)
    assert C.exp_inside(T1).all() and C.exp_inside(lam * S).all()
