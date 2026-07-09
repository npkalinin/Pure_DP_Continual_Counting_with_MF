"""Verification of the floating-point bound from Proposition 2.6."""

import numpy as np
import sympy as sp
from math import isqrt

K = 1024
DENOM = 2 ** 32
PRECISION_BITS = 128

TARGETS = {
    "maxse": {"L_tilde": "1.8691", "delta_L": "0.0039", "L": "1.8730", "R": "3.8417"},
    "meanse": {"L_tilde": "15.3661", "delta_L": "0.0037", "L": "15.3661", "R": "14.3098"},
}


def rational_sqrt_upper_bound(x, precision_bits=PRECISION_BITS):
    x = sp.Rational(x)
    scale = 1 << precision_bits
    a, b = x.as_numer_denom()
    a, b = int(a), int(b)
    y = isqrt((a * scale ** 2) // b) + 1
    sqrt_upper_bound = sp.Rational(y, scale)
    assert x < sqrt_upper_bound ** 2
    return sqrt_upper_bound


def frobenius_norm_upper_bound(M):
    return rational_sqrt_upper_bound(sum(x ** 2 for x in M))


def max_row_l2_norm_upper_bound(M):
    row_sq_norms = [sum(M[i, j] ** 2 for j in range(M.cols)) for i in range(M.rows)]
    return rational_sqrt_upper_bound(max(row_sq_norms))


def max_col_l1_norm(M):
    return max(sum(abs(M[i, j]) for i in range(M.rows)) for j in range(M.cols))


def verify(error_type):
    R_float = np.load(f"{error_type}_n{K}_R.npy").astype("float64")
    T_k = np.tri(K, dtype="int64")
    T_k_inv = np.linalg.inv(T_k).astype("int64")
    L_float = T_k @ np.linalg.inv(R_float)

    R_int = np.round(R_float * DENOM).astype(np.int64)
    R_rational = sp.Matrix(R_int).applyfunc(lambda x: sp.Rational(int(x), DENOM))

    L_int = np.round(L_float * DENOM).astype(np.int64)
    L_rational = sp.Matrix(L_int).applyfunc(lambda x: sp.Rational(int(x), DENOM))

    T_k_rational = sp.Matrix(T_k)
    T_k_inv_rational = sp.Matrix(T_k_inv)

    R_column_norm = max_col_l1_norm(R_rational)
    assert R_column_norm <= sp.Rational(TARGETS[error_type]["R"])

    if error_type == "maxse":
        L_tilde_row_norm = max_row_l2_norm_upper_bound(L_rational)
    else:
        L_tilde_row_norm = frobenius_norm_upper_bound(L_rational)
    assert L_tilde_row_norm <= sp.Rational(TARGETS[error_type]["L_tilde"])

    Delta_rational = L_rational @ R_rational - T_k_rational
    q = frobenius_norm_upper_bound(T_k_inv_rational @ Delta_rational)
    assert q < 1  # required for the Neumann series argument

    T_k_norm_F = frobenius_norm_upper_bound(T_k_rational)
    Tinv_L_rational_norm_F = frobenius_norm_upper_bound(T_k_inv_rational @ L_rational)

    delta_L_norm = (T_k_norm_F * Tinv_L_rational_norm_F * q) / (1 - q)
    assert delta_L_norm <= sp.Rational(TARGETS[error_type]["delta_L"])

    L_norm = L_tilde_row_norm + delta_L_norm
    assert L_norm <= sp.Rational(TARGETS[error_type]["L"])

    print(f"{error_type}: norms from the proposition verified successfully")


if __name__ == "__main__":
    verify("maxse")
    verify("meanse")
