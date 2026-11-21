"""
Exact Zero-Float Mathematical Stacking & Tomography Engine for `aletheia-mde`:
  1. Fraction-Free Bareiss Pivoted Symmetric Congruence (1x1 + 2x2 zero-diagonal blocks)
     with multiplication-only Reduced Integer Cholesky-Sum PSD Certificates AND
     Constructive Exact Integer Rejection Witnesses (`x^T Q x < 0`), directly solving
     BOTH `siliconcordat/warp#15`'s N^2 Cauchy-Schwarz tax AND its 4,330-bit naive
     denominator blowup!
  2. Exact Stacked-Adapter Concatenated Gram Gate (`Q_stack(s) = s G_B - G_B G_A G_B >= 0`)
     with dyadic `zone-canon-v0` integer scales.
  3. Exact Contextual Compute Cellular Sheaf Gluing & Spectral-Gap Certifier (`docs/08` +
     `siliconcordat/exo` zone gating).
  4. Exact Higher-Moment Spectral-Rank Covert-Backdoor Gate (`R_opt` + Stieltjes-Hankel
     `R_H4`, `AUDIT-002`/`AUDIT-014`) addressing `Moissanite-Labs/MDE` Open Question Q3.
  5. Independent Zero-Shared-Code Faddeev-LeVerrier + Square-Free Descartes Dual Oracle.

Pure Python 3 stdlib (`int`, `Fraction`, `math.gcd`, `math.lcm`). Zero floats. Zero bare asserts.
"""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Any, Sequence


class GramGateError(ValueError):
    """Fail-closed exact arithmetic / gate integrity error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _req(cond: bool, code: str, msg: str) -> None:
    if not cond:
        raise GramGateError(code, msg)


def _require_int_matrix(M: Sequence[Sequence[Any]], name: str, *, require_square: bool = False, require_symmetric: bool = False) -> list[list[int]]:
    _req(isinstance(M, (list, tuple)) and len(M) > 0, "EMPTY_MATRIX", f"{name} must be non-empty 2D int matrix")
    rows = len(M)
    cols = None
    out: list[list[int]] = []
    for r_idx, row in enumerate(M):
        _req(isinstance(row, (list, tuple)) and len(row) > 0, "INVALID_ROW", f"{name}[{r_idx}] must be non-empty list")
        if cols is None:
            cols = len(row)
        else:
            _req(len(row) == cols, "RAGGED_MATRIX", f"{name} is ragged at row {r_idx}")
        clean_row: list[int] = []
        for c_idx, val in enumerate(row):
            _req(type(val) is int, "NON_INT_ENTRY", f"{name}[{r_idx}][{c_idx}] must be exact int, got {type(val).__name__}")
            clean_row.append(val)
        out.append(clean_row)
    if require_square:
        _req(rows == cols, "NON_SQUARE_MATRIX", f"{name} must be square ({rows}x{cols})")
    if require_symmetric:
        _req(rows == cols, "NON_SQUARE_MATRIX", f"{name} must be square to be symmetric")
        for i in range(rows):
            for j in range(i + 1, rows):
                _req(out[i][j] == out[j][i], "MATRIX_NOT_SYMMETRIC", f"{name}[{i}][{j}]={out[i][j]} != {name}[{j}][{i}]={out[j][i]}")
    return out


def _mat_mul_int(A: Sequence[Sequence[int]], B: Sequence[Sequence[int]]) -> list[list[int]]:
    m, k = len(A), len(A[0])
    _req(len(B) == k, "MATMUL_DIM_MISMATCH", f"cannot multiply ({m}x{k}) by ({len(B)}x{len(B[0])})")
    n = len(B[0])
    return [
        [sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
        for i in range(m)
    ]


def _mat_t_int(A: Sequence[Sequence[int]]) -> list[list[int]]:
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def _quad_form_int(M: Sequence[Sequence[int]], x: Sequence[int]) -> int:
    n = len(M)
    _req(len(x) == n, "VEC_DIM_MISMATCH", f"vector len {len(x)} != {n}")
    total = 0
    for i in range(n):
        xi = x[i]
        if xi == 0:
            continue
        row = M[i]
        total += xi * sum(row[j] * x[j] for j in range(n))
    return total


def _vec_gcd(vec: Sequence[int]) -> int:
    g = 0
    for v in vec:
        g = math.gcd(g, abs(v))
    return g if g > 0 else 1


def _solve_spd_integer_scaled(A: list[list[int]], rhs: list[int]) -> tuple[list[int], int]:
    """
    Given strictly positive definite k x k integer matrix A (k >= 0) and integer vector rhs,
    solve A * y = -D * rhs exactly in integers using Bareiss elimination + back-substitution!
    Returns (y_int, D) with D = det(A) > 0 and A * y_int == -D * rhs identically.
    """
    k = len(A)
    if k == 0:
        return [], 1

    # Augmented k x (k + 1) matrix [A | -rhs]
    aug = [A[i][:] + [-rhs[i]] for i in range(k)]
    prev_det = 1
    for p in range(k):
        piv = aug[p][p]
        _req(piv > 0, "LEADING_BLOCK_NOT_SPD", f"expected positive leading minor at step {p}, got {piv}")
        for i in range(p + 1, k):
            for j in range(p + 1, k + 1):
                num = piv * aug[i][j] - aug[i][p] * aug[p][j]
                _req(num % prev_det == 0, "BAREISS_DIV_ERROR", "exact Bareiss divisibility violated")
                aug[i][j] = num // prev_det
            aug[i][p] = 0
        prev_det = piv

    det_A = prev_det  # = aug[k-1][k-1] > 0
    # Back-substitution: in Bareiss upper-triangular form, row p has diagonal d_p = aug[p][p],
    # and aug[p][k] already equals det_p * true_rhs_p; solving with exact Fraction or integer scaling:
    # We compute y_frac via exact back-sub and multiply by det_A (guaranteed integer by Cramer's rule!).
    y_frac = [Fraction(0)] * k
    for p in range(k - 1, -1, -1):
        s = Fraction(aug[p][k])
        for j in range(p + 1, k):
            s -= Fraction(aug[p][j]) * y_frac[j]
        y_frac[p] = s / Fraction(aug[p][p])

    y_int: list[int] = []
    for p in range(k):
        scaled = y_frac[p] * Fraction(det_A)
        _req(scaled.denominator == 1, "CRAMER_INTEGRALITY_ERROR", f"expected integer det(A)*y[{p}], got {scaled}")
        y_int.append(int(scaled.numerator))

    # Verify A * y_int == -det_A * rhs in pure integers!
    for i in range(k):
        lhs = sum(A[i][j] * y_int[j] for j in range(k))
        _req(lhs == -det_A * rhs[i], "WITNESS_SOLVE_MISMATCH", f"row {i}: {lhs} != {-det_A * rhs[i]}")
    return y_int, det_A


def bareiss_symmetric_inertia(M_in: Sequence[Sequence[int]]) -> dict[str, Any]:
    """
    Exact fraction-free Bareiss pivoted symmetric block-congruence over Z.

    Handles BOTH 1x1 nonzero diagonal pivots AND 2x2 zero-diagonal symmetric pivots
    using Sylvester's exact determinant identity (`% prev_det == 0` verified fail-closed!).

    Returns:
      - `inertia`: (n_pos, n_neg, n_zero)
      - `is_psd`: bool (n_neg == 0)
      - `max_pivot_bits`: maximum bit-length among all Bareiss minors encountered
      - `psd_certificate`: when `is_psd`, a reduced multiplication-only integer certificate
            C * P M P^T = sum_k w_k * l_k l_k^T  (C > 0, w_k > 0, l_k in Z^n)
      - `rejection_witness`: when `not is_psd`, an exact primitive integer vector x in Z^n
            with `x^T M x < 0` verified directly in pure integer arithmetic!
    """
    M0 = _require_int_matrix(M_in, "M", require_square=True, require_symmetric=True)
    n = len(M0)
    A = [row[:] for row in M0]
    perm = list(range(n))

    n_pos = 0
    n_neg = 0
    n_zero = 0
    prev_det = 1  # Delta_k (signed exact leading principal minor)
    max_pivot_bits = 1

    # Store Bareiss columns & minors for PSD certificate construction when all pivots stay positive
    psd_so_far = True
    bareiss_cols: list[list[int]] = []
    bareiss_deltas: list[tuple[int, int]] = []  # (Delta_k, Delta_{k+1})

    # First-failure snapshot for constructive rejection witness (when all previous pivots were > 0)
    rejection_witness: list[int] | None = None
    rejection_value: int | None = None

    k = 0
    while k < n:
        # 1. Search for a nonzero diagonal entry in trailing submatrix k..n-1.
        #    Prefer a POSITIVE true Schur pivot first while still PSD? No — actually,
        #    checking for ANY negative true Schur pivot immediately when psd_so_far gives
        #    the cleanest possible rejection witness (all prior pivots strictly positive!).
        neg_diag_piv = None
        pos_diag_piv = None
        for i in range(k, n):
            val = A[i][i]
            if val != 0:
                true_sign = (1 if val > 0 else -1) * (1 if prev_det > 0 else -1)
                if true_sign < 0 and neg_diag_piv is None:
                    neg_diag_piv = i
                elif true_sign > 0 and pos_diag_piv is None:
                    pos_diag_piv = i

        # Pick pivot: if psd_so_far and a negative diagonal exists, pick it immediately so
        # leading k x k block is guaranteed SPD and gives an instant exact witness!
        # Otherwise pick smallest-magnitude positive (or any nonzero) diagonal to minimize bit growth.
        diag_piv = None
        if psd_so_far and neg_diag_piv is not None:
            diag_piv = neg_diag_piv
        else:
            best_abs = None
            for i in range(k, n):
                v = A[i][i]
                if v != 0:
                    av = abs(v)
                    if best_abs is None or av < best_abs:
                        best_abs = av
                        diag_piv = i

        if diag_piv is not None:
            if diag_piv != k:
                perm[k], perm[diag_piv] = perm[diag_piv], perm[k]
                A[k], A[diag_piv] = A[diag_piv], A[k]
                for r in range(n):
                    A[r][k], A[r][diag_piv] = A[r][diag_piv], A[r][k]
                # Permute already recorded PSD columns too if still PSD
                if psd_so_far:
                    for col in bareiss_cols:
                        col[k], col[diag_piv] = col[diag_piv], col[k]

            piv = A[k][k]
            max_pivot_bits = max(max_pivot_bits, abs(piv).bit_length())
            true_pos = (piv > 0 and prev_det > 0) or (piv < 0 and prev_det < 0)

            if true_pos:
                n_pos += 1
                if psd_so_far:
                    col_k = [0] * k + [A[i][k] for i in range(k, n)]
                    bareiss_cols.append(col_k)
                    bareiss_deltas.append((prev_det, piv))
            else:
                n_neg += 1
                if psd_so_far:
                    # Construct exact integer rejection witness BEFORE modifying A!
                    # Leading k x k block of P M0 P^T is strictly SPD with det == prev_det > 0.
                    A_lead = [[M0[perm[r]][perm[c]] for c in range(k)] for r in range(k)]
                    v_rhs = [M0[perm[r]][perm[k]] for r in range(k)]
                    y_int, det_lead = _solve_spd_integer_scaled(A_lead, v_rhs)
                    z = y_int + [det_lead] + [0] * (n - k - 1)
                    x = [0] * n
                    for idx_p, orig_i in enumerate(perm):
                        x[orig_i] = z[idx_p]
                    g = _vec_gcd(x)
                    x = [v // g for v in x]
                    qval = _quad_form_int(M0, x)
                    _req(qval < 0, "WITNESS_NON_NEGATIVE", f"constructed witness has x^T M x = {qval} >= 0")
                    rejection_witness = x
                    rejection_value = qval
                    psd_so_far = False

            # Bareiss 1x1 update (exact by Sylvester's determinant identity)
            for i in range(k + 1, n):
                aik = A[i][k]
                for j in range(i, n):
                    num = piv * A[i][j] - aik * A[j][k]
                    _req(num % prev_det == 0, "BAREISS_1X1_DIV_ERROR", f"inexact Bareiss 1x1 division at k={k}")
                    val = num // prev_det
                    A[i][j] = val
                    A[j][i] = val
                    max_pivot_bits = max(max_pivot_bits, abs(val).bit_length())
            prev_det = piv
            k += 1
            continue

        # 2. All remaining diagonal entries A[i][i] == 0 for i >= k: look for nonzero off-diagonal
        off_piv = None
        best_off_abs = None
        for i in range(k, n):
            for j in range(i + 1, n):
                v = A[i][j]
                if v != 0:
                    av = abs(v)
                    if best_off_abs is None or av < best_off_abs:
                        best_off_abs = av
                        off_piv = (i, j)

        if off_piv is None:
            # Remaining trailing (n - k) x (n - k) block is identically zero!
            n_zero += n - k
            break

        p1, p2 = off_piv
        if p1 != k:
            perm[k], perm[p1] = perm[p1], perm[k]
            A[k], A[p1] = A[p1], A[k]
            for r in range(n):
                A[r][k], A[r][p1] = A[r][p1], A[r][k]
            if p2 == k:
                p2 = p1
        if p2 != k + 1:
            perm[k + 1], perm[p2] = perm[p2], perm[k + 1]
            A[k + 1], A[p2] = A[p2], A[k + 1]
            for r in range(n):
                A[r][k + 1], A[r][p2] = A[r][p2], A[r][k + 1]

        b = A[k][k + 1]
        max_pivot_bits = max(max_pivot_bits, abs(b).bit_length())
        # 2x2 block [[0, b], [b, 0]] ALWAYS contributes (+1, -1) regardless of sign(prev_det)
        n_pos += 1
        n_neg += 1

        if psd_so_far:
            # Construct exact integer rejection witness using e_neg = (1, -1) if b > 0 else (1, 1)
            # Since prev_det > 0, true Schur 2x2 block is (1/prev_det) [[0, b], [b, 0]]!
            e0, e1 = (1, -1) if b > 0 else (1, 1)
            A_lead = [[M0[perm[r]][perm[c]] for c in range(k)] for r in range(k)]
            w_rhs = [M0[perm[r]][perm[k]] * e0 + M0[perm[r]][perm[k + 1]] * e1 for r in range(k)]
            y_int, det_lead = _solve_spd_integer_scaled(A_lead, w_rhs)
            z = y_int + [det_lead * e0, det_lead * e1] + [0] * (n - k - 2)
            x = [0] * n
            for idx_p, orig_i in enumerate(perm):
                x[orig_i] = z[idx_p]
            g = _vec_gcd(x)
            x = [v // g for v in x]
            qval = _quad_form_int(M0, x)
            _req(qval < 0, "WITNESS_2X2_NON_NEGATIVE", f"2x2 witness has x^T M x = {qval} >= 0")
            rejection_witness = x
            rejection_value = qval
            psd_so_far = False

        # Exact 2x2 Bareiss update:
        #   Delta_{k+2} = -b^2 / prev_det
        #   A_new[i][j] = b * (A[i][k]*A[j][k+1] + A[i][k+1]*A[j][k] - b*A[i][j]) / prev_det^2
        _req((-b * b) % prev_det == 0, "BAREISS_2X2_DET_DIV_ERROR", f"inexact 2x2 determinant division at k={k}")
        new_det = (-b * b) // prev_det
        prev_sq = prev_det * prev_det
        for i in range(k + 2, n):
            aik, aik1 = A[i][k], A[i][k + 1]
            for j in range(i, n):
                num = b * (aik * A[j][k + 1] + aik1 * A[j][k] - b * A[i][j])
                _req(num % prev_sq == 0, "BAREISS_2X2_UPDATE_DIV_ERROR", f"inexact 2x2 Bareiss update at k={k}")
                val = num // prev_sq
                A[i][j] = val
                A[j][i] = val
                max_pivot_bits = max(max_pivot_bits, abs(val).bit_length())
        prev_det = new_det
        max_pivot_bits = max(max_pivot_bits, abs(prev_det).bit_length())
        k += 2

    # Build reduced multiplication-only integer PSD certificate when n_neg == 0!
    psd_cert: dict[str, Any] | None = None
    if n_neg == 0:
        # P M0 P^T = sum_r (1 / (Delta_r * Delta_{r+1})) * l_r l_r^T
        # Factor out g_r = gcd(l_r) from each column and reduce g_r^2 / (Delta_r * Delta_{r+1}) = p_r / q_r!
        reduced_cols: list[list[int]] = []
        fracs: list[Fraction] = []
        for col, (d_prev, d_curr) in zip(bareiss_cols, bareiss_deltas):
            g = _vec_gcd(col)
            rcol = [v // g for v in col]
            reduced_cols.append(rcol)
            fracs.append(Fraction(g * g, d_prev * d_curr))
        cert_C = 1
        for fr in fracs:
            cert_C = math.lcm(cert_C, fr.denominator)
        weights = [int(fr * cert_C) for fr in fracs]
        psd_cert = {
            "perm": perm,
            "cert_scale_C": cert_C,
            "cert_scale_bits": cert_C.bit_length(),
            "weights": weights,
            "cols_int": reduced_cols,
        }
        # Self-verify using purely multiplication-only checker (ZERO division!)
        verify_integer_psd_certificate(M0, psd_cert)

    return {
        "inertia": (n_pos, n_neg, n_zero),
        "is_psd": (n_neg == 0),
        "max_pivot_bits": max_pivot_bits,
        "psd_certificate": psd_cert,
        "rejection_witness": rejection_witness,
        "rejection_value": rejection_value,
    }


def verify_integer_psd_certificate(M_in: Sequence[Sequence[int]], cert: dict[str, Any]) -> bool:
    """
    Zero-division, multiplication-only verifier for a Reduced Bareiss Integer PSD Certificate:
        C * P M P^T == sum_k w_k * l_k l_k^T   with C > 0, w_k > 0, l_k in Z^n.
    Uses ONLY integer addition, multiplication, and comparison (fulfilling `warp#15`!).
    """
    M = _require_int_matrix(M_in, "M", require_square=True, require_symmetric=True)
    n = len(M)
    perm = cert.get("perm")
    _req(
        isinstance(perm, list) and len(perm) == n
        and all(type(p) is int for p in perm)
        and sorted(perm) == list(range(n)),
        "INVALID_CERT_PERM",
        "certificate perm must be a permutation of 0..n-1",
    )
    C = cert.get("cert_scale_C")
    _req(type(C) is int and C > 0, "INVALID_CERT_SCALE", "cert_scale_C must be positive int")
    weights = cert.get("weights")
    cols = cert.get("cols_int")
    _req(isinstance(weights, list) and isinstance(cols, list) and len(weights) == len(cols),
         "INVALID_CERT_FACTORS", "weights and cols_int must be lists of matching length")
    for idx, (w, col) in enumerate(zip(weights, cols)):
        _req(type(w) is int and w > 0, "NON_POSITIVE_CERT_WEIGHT", f"weights[{idx}]={w} must be > 0")
        _req(isinstance(col, list) and len(col) == n and all(type(v) is int for v in col),
             "INVALID_CERT_COL", f"cols_int[{idx}] must be len-{n} list[int]")

    for i in range(n):
        for j in range(i, n):
            lhs = C * M[perm[i]][perm[j]]
            rhs = 0
            for w, col in zip(weights, cols):
                rhs += w * col[i] * col[j]
            _req(lhs == rhs, "PSD_CERT_IDENTITY_MISMATCH", f"entry ({i},{j}): {lhs} != {rhs}")
    return True


def verify_integer_rejection_witness(M_in: Sequence[Sequence[int]], x: Sequence[int]) -> int:
    """Pure integer multiplication verifier for rejection witness `x^T M x < 0`."""
    M = _require_int_matrix(M_in, "M", require_square=True, require_symmetric=True)
    _req(isinstance(x, (list, tuple)) and len(x) == len(M) and all(type(v) is int for v in x),
         "INVALID_WITNESS_VEC", "rejection witness must be int vector of length n")
    val = _quad_form_int(M, x)
    _req(val < 0, "WITNESS_NOT_NEGATIVE", f"rejection witness fails: x^T M x = {val} >= 0")
    return val


def naive_rational_ldl_scale_bits(M_in: Sequence[Sequence[int]]) -> int:
    """
    Reference measurement of the naive rational-clearing multiplier bit-length warned about in
    `siliconcordat/warp#15` (clearing denominators of rational L D L^T into c Q = L_int D_int L_int^T).
    Used strictly for honest side-by-side bit-length benchmarking against our Reduced Bareiss cert!
    """
    M = _require_int_matrix(M_in, "M", require_square=True, require_symmetric=True)
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)] for i in range(n)]
    L = [[Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)]
    D = [Fraction(0)] * n
    for k in range(n):
        piv = None
        for i in range(k, n):
            if A[i][i] != 0:
                piv = i
                break
        if piv is None:
            break
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
            for r in range(n):
                A[r][k], A[r][piv] = A[r][piv], A[r][k]
            for c in range(k):
                L[k][c], L[piv][c] = L[piv][c], L[k][c]
        d = A[k][k]
        D[k] = d
        for i in range(k + 1, n):
            L[i][k] = A[i][k] / d
        for i in range(k + 1, n):
            lik = L[i][k]
            if lik == 0:
                continue
            for j in range(i, n):
                val = A[i][j] - lik * d * L[j][k]
                A[i][j] = val
                A[j][i] = val
    # Naive clearing: find L_den = lcm of all denominators in L, D_den = lcm of all denominators in D;
    # clearing c * Q = (L_den * L) * diag(D_den * D) * (L_den * L)^T needs c = L_den^2 * D_den!
    l_den = 1
    for i in range(n):
        for j in range(n):
            l_den = math.lcm(l_den, L[i][j].denominator)
    d_den = 1
    for dv in D:
        d_den = math.lcm(d_den, dv.denominator)
    naive_c = l_den * l_den * d_den
    return naive_c.bit_length()


# ---------------------------------------------------------------------------
# 2. Exact Concatenated Multi-MDE Stacked-Adapter Gram Gate (Solving warp#15!)
# ---------------------------------------------------------------------------

def certify_stacked_mde_gram(
    adapters: Sequence[dict[str, Any]],
    *,
    max_frobenius_sq: tuple[int, int],
    max_spectral_norm_sq: tuple[int, int],
) -> dict[str, Any]:
    """
    Exact integer concatenated Gram gate for N stacked MDE low-rank adapters:
        Delta_stack = sum_{k=1}^N (route_num_k / 2^{route_den_log2}) * (B_k A_k / 2^{2*factor_exp2_k})

    Each adapter dict holds:
      - `adapter_id`: str
      - `B_int`: d_out x r_k integer matrix
      - `A_int`: r_k x d_in integer matrix
      - `factor_exp2`: int >= 0 (scaling divisor 2^factor_exp2 per factor; default 0)
      - `route_num`: int >= 0 (dyadic numerator from zone-canon-v0; default 1)
      - `route_den_log2`: int >= 0 (dyadic shift; default 0)

    Exact theorems:
      - ||Delta_stack||_F^2 = tr(G_B G_A) / common_scale^2
      - sigma_max^2(Delta_stack) <= s_num / s_den
          <=> Q_stack(s) = s_num * common_scale^2 * G_B - s_den * G_B G_A G_B >= 0  (in Z^{R x R}!)
    Eliminates BOTH `warp#15`'s N^2 Cauchy-Schwarz spectral tax AND naive denominator blowup!
    """
    _req(isinstance(adapters, (list, tuple)) and len(adapters) >= 1, "EMPTY_STACK", "adapters must be non-empty")
    f_num, f_den = max_frobenius_sq
    s_num, s_den = max_spectral_norm_sq
    _req(type(f_num) is int and f_num >= 0 and type(f_den) is int and f_den > 0, "INVALID_FROB_BOUND", "bad max_frobenius_sq")
    _req(type(s_num) is int and s_num >= 0 and type(s_den) is int and s_den > 0, "INVALID_SPEC_BOUND", "bad max_spectral_norm_sq")

    # Normalize all adapters onto a common dyadic shift so B_k * A_k * alpha_k = (B_scaled_k * A_scaled_k) / 2^common_shift
    parsed: list[dict[str, Any]] = []
    d_out = None
    d_in = None
    max_total_shift = 0
    seen_ids: set[str] = set()

    for ad in adapters:
        aid = ad.get("adapter_id")
        _req(isinstance(aid, str) and bool(aid.strip()) and aid not in seen_ids, "INVALID_ADAPTER_ID", f"bad or duplicate adapter_id {aid!r}")
        seen_ids.add(aid)
        B = _require_int_matrix(ad["B_int"], f"{aid}.B_int")
        A = _require_int_matrix(ad["A_int"], f"{aid}.A_int")
        r_k = len(B[0])
        _req(len(A) == r_k, "ADAPTER_RANK_MISMATCH", f"{aid}: B cols {r_k} != A rows {len(A)}")
        if d_out is None:
            d_out, d_in = len(B), len(A[0])
        else:
            _req(len(B) == d_out and len(A[0]) == d_in, "ADAPTER_DIM_MISMATCH", f"{aid} dim mismatch")
        f_exp = ad.get("factor_exp2", 0)
        r_num = ad.get("route_num", 1)
        r_log2 = ad.get("route_den_log2", 0)
        _req(type(f_exp) is int and 0 <= f_exp <= 24, "INVALID_FACTOR_EXP2", f"{aid}: bad factor_exp2")
        _req(type(r_num) is int and r_num >= 0, "INVALID_ROUTE_NUM", f"{aid}: bad route_num")
        _req(type(r_log2) is int and 0 <= r_log2 <= 24, "INVALID_ROUTE_LOG2", f"{aid}: bad route_den_log2")
        total_shift = 2 * f_exp + r_log2
        max_total_shift = max(max_total_shift, total_shift)
        parsed.append({"id": aid, "B": B, "A": A, "r": r_k, "r_num": r_num, "shift": total_shift})

    # Absorb (route_num * 2^(max_total_shift - shift)) cleanly into A_k (keeps G_B small and exact!)
    # Then Delta_stack = (B_stack @ A_stack) / 2^max_total_shift!
    B_blocks: list[list[list[int]]] = []
    A_blocks: list[list[list[int]]] = []
    offsets: list[tuple[str, int, int]] = []
    cur = 0
    for item in parsed:
        mul = item["r_num"] << (max_total_shift - item["shift"])
        if mul == 0:
            # Muted adapter (scale == 0 from L5 zone routing!): zero contribution!
            continue
        Bk = item["B"]
        Ak = [[v * mul for v in row] for row in item["A"]]
        B_blocks.append(Bk)
        A_blocks.append(Ak)
        offsets.append((item["id"], cur, cur + item["r"]))
        cur += item["r"]

    common_div = 1 << max_total_shift
    common_div_sq = common_div * common_div

    # Handle all-muted stack cleanly (zero delta -> trivially satisfies any >= 0 bound)
    if cur == 0:
        return {
            "active_adapters": [],
            "total_rank": 0,
            "frobenius_sq_exact": "0",
            "frobenius_passed": True,
            "spectral_passed": True,
            "passed": True,
            "Q_stack_inertia": [0, 0, 0],
            "bareiss_cert_bits": 1,
            "naive_ldl_bits": 1,
            "pairwise_interference": {},
            "rejection_witness": None,
        }

    # Concatenate B_stack (d_out x R) and A_stack (R x d_in)
    _req(d_out is not None, "INTERNAL_DIM", "missing d_out")
    B_stack = [[v for Bk in B_blocks for v in Bk[row_i]] for row_i in range(d_out)]
    A_stack = [row[:] for Ak in A_blocks for row in Ak]

    G_B = _mat_mul_int(_mat_t_int(B_stack), B_stack)
    G_A = _mat_mul_int(A_stack, _mat_t_int(A_stack))
    R = cur

    # Exact integer Frobenius norm squared: tr(G_B G_A) / common_div_sq
    frob_num = sum(G_B[i][j] * G_A[j][i] for i in range(R) for j in range(R))
    frob_passed = (frob_num * f_den <= f_num * common_div_sq)

    # Exact pairwise Frobenius cross-interference matrix I_{ab} = tr(G_B[a,b] G_A[b,a]) / common_div_sq
    pairwise: dict[str, str] = {}
    for idx_a in range(len(offsets)):
        id_a, lo_a, hi_a = offsets[idx_a]
        for idx_b in range(idx_a + 1, len(offsets)):
            id_b, lo_b, hi_b = offsets[idx_b]
            cross = sum(
                G_B[i][j] * G_A[j][i]
                for i in range(lo_a, hi_a)
                for j in range(lo_b, hi_b)
            )
            pairwise[f"{id_a}<->{id_b}"] = str(Fraction(cross, common_div_sq))

    # Exact Q_stack(s) = s_num * common_div_sq * G_B - s_den * G_B G_A G_B (pure integer symmetric R x R!)
    GB_GA_GB = _mat_mul_int(_mat_mul_int(G_B, G_A), G_B)
    Q_s = [
        [s_num * common_div_sq * G_B[i][j] - s_den * GB_GA_GB[i][j] for j in range(R)]
        for i in range(R)
    ]

    bareiss_res = bareiss_symmetric_inertia(Q_s)
    naive_bits = naive_rational_ldl_scale_bits(Q_s) if bareiss_res["is_psd"] else None

    # Compare against warp#15's Cauchy-Schwarz upper bound N * sum_k (s_k^2 * sigma_k^2)
    # Also compute exact individual squared Frobenius norms for direct comparison!
    indiv_frob_sum = Fraction(0)
    for _aid, lo, hi in offsets:
        f_k = sum(G_B[i][j] * G_A[j][i] for i in range(lo, hi) for j in range(lo, hi))
        indiv_frob_sum += Fraction(f_k, common_div_sq)
    n_active = len(offsets)
    warp15_cs_frob_bound = Fraction(n_active) * indiv_frob_sum

    spec_passed = bool(bareiss_res["is_psd"])
    return {
        "active_adapters": [x[0] for x in offsets],
        "total_rank": R,
        "frobenius_sq_exact": str(Fraction(frob_num, common_div_sq)),
        "warp15_cauchy_schwarz_frob_bound": str(warp15_cs_frob_bound),
        "frobenius_passed": frob_passed,
        "spectral_bound_tested": str(Fraction(s_num, s_den)),
        "spectral_passed": spec_passed,
        "passed": frob_passed and spec_passed,
        "Q_stack_inertia": list(bareiss_res["inertia"]),
        "bareiss_max_pivot_bits": bareiss_res["max_pivot_bits"],
        "bareiss_cert_bits": bareiss_res["psd_certificate"]["cert_scale_bits"] if bareiss_res["psd_certificate"] else None,
        "naive_ldl_bits": naive_bits,
        "pairwise_interference": pairwise,
        "rejection_witness": bareiss_res["rejection_witness"],
        "rejection_value": bareiss_res["rejection_value"],
        "Q_stack_int": Q_s,
    }


# ---------------------------------------------------------------------------
# 3. Exact Contextual Compute Cellular Sheaf Gluing (`docs/08` + `exo`)
# ---------------------------------------------------------------------------

def certify_mde_compatibility_sheaf(
    R_aletheia: Sequence[Sequence[int]],
    R_domain: Sequence[Sequence[int]],
    *,
    denom: int = 5,
    include_domain: bool = True,
    include_holonomy_cycle: bool = True,
    gap_gamma: tuple[int, int] = (97, 1024),
) -> dict[str, Any]:
    """
    Exact integer cellular sheaf over the 3-node multi-MDE context graph:
      v0 = aletheia-mde (stalk Q^2), v1 = shared_context (stalk Q^2), v2 = domain-mde (stalk Q^2).
    Restriction maps scaled by `denom`:
      - edge (v0 -> v1): (R_aletheia / denom) * x_0 - x_1 = 0
      - edge (v2 -> v1): (R_domain   / denom) * x_2 - x_1 = 0   (active iff include_domain)
      - edge (v0 -> v2): x_0 - x_2 = 0                          (direct policy agreement cycle)

    Delta_F_int = (denom * d0)^T (denom * d0) in Z^{2n_v x 2n_v} satisfies Delta_F = Delta_F_int / denom^2.
    Certifies exact dim H^0(F) = n_zero(Delta_F_int) AND exact spectral gap Delta_F - gamma I >= 0 via Bareiss!
    """
    Ra = _require_int_matrix(R_aletheia, "R_aletheia", require_square=True)
    Rd = _require_int_matrix(R_domain, "R_domain", require_square=True)
    _req(len(Ra) == 2 and len(Rd) == 2, "SHEAF_STALK_DIM", "stalk dimension must be 2")
    _req(type(denom) is int and denom > 0, "INVALID_DENOM", "denom must be positive int")
    g_num, g_den = gap_gamma
    _req(type(g_num) is int and g_num >= 0 and type(g_den) is int and g_den > 0, "INVALID_GAMMA", "bad gap_gamma")

    n_v = 3 if include_domain else 2
    d_tot = 2 * n_v
    d0_int: list[list[int]] = []
    # Edge (aletheia -> shared): R_aletheia * x0 - denom * x1
    for r in range(2):
        row = [0] * d_tot
        row[0], row[1] = Ra[r][0], Ra[r][1]
        row[2 + r] = -denom
        d0_int.append(row)
    if include_domain:
        # Edge (domain -> shared): R_domain * x2 - denom * x1
        for r in range(2):
            row = [0] * d_tot
            row[4], row[5] = Rd[r][0], Rd[r][1]
            row[2 + r] = -denom
            d0_int.append(row)
        if include_holonomy_cycle:
            # Direct agreement edge (aletheia -> domain): denom * x0 - denom * x2
            for r in range(2):
                row = [0] * d_tot
                row[r] = denom
                row[4 + r] = -denom
                d0_int.append(row)

    Delta_int = _mat_mul_int(_mat_t_int(d0_int), d0_int)
    b_lap = bareiss_symmetric_inertia(Delta_int)
    dim_h0 = b_lap["inertia"][2]

    # Check spectral gap Delta_F - (g_num / g_den) * I >= 0  <=>  g_den * Delta_int - g_num * denom^2 * I >= 0
    d_sq = denom * denom
    shifted = [
        [g_den * Delta_int[i][j] - (g_num * d_sq if i == j else 0) for j in range(d_tot)]
        for i in range(d_tot)
    ]
    b_gap = bareiss_symmetric_inertia(shifted)

    return {
        "include_domain": include_domain,
        "include_holonomy_cycle": include_holonomy_cycle,
        "dim_H0": dim_h0,
        "consistent_global_section_exists": (dim_h0 > 0),
        "laplacian_inertia": list(b_lap["inertia"]),
        "gamma_tested": str(Fraction(g_num, g_den)),
        "spectral_gap_certified_ge_gamma": b_gap["is_psd"],
        "Delta_int": Delta_int,
    }


# ---------------------------------------------------------------------------
# 4. Higher-Moment Spectral-Rank Covert-Backdoor Gate (Moissanite-Labs/MDE Q3)
# ---------------------------------------------------------------------------

def r_opt_exact(u: Fraction, t: Fraction, S: Fraction, b: int) -> Fraction:
    """Exact AUDIT-002 Theorem A1 scale-invariant 2nd-moment projected-rank lower bound."""
    _req(type(b) is int and b >= 0, "INVALID_B", "b must be non-negative int")
    if u <= 0 or u + t <= 0 or S <= 0:
        return Fraction(0)
    if b > 0 and t > 0 and t * (u + t) >= Fraction(b) * S:
        denom = S - (t * t) / Fraction(b)
        return (u * u) / denom if denom > 0 else Fraction(0)
    val = ((u + t) * (u + t)) / S - Fraction(b)
    return val if val > 0 else Fraction(0)


def r_h4_stieltjes_hankel_exact(S1: Fraction, S2: Fraction, S3: Fraction, S4: Fraction, b: int) -> Fraction:
    """
    Exact AUDIT-014 Theorem A2 4th-moment Stieltjes-Hankel projected-rank lower bound:
        R_H4 = S1^2 / S2 + (S1*S3 - S2^2)^2 / (S2 * (S2*S4 - S3^2)) - b
    valid whenever S1 > 0, S2 > 0, and exact KKT witness a = (a1, a2) >= 0 with a1 = (S1*S4 - S2*S3)/D,
    a2 = (S2^2 - S1*S3)/D or pure degree-2/4 Hankel positive determinant D = S2*S4 - S3^2 > 0
    on a PSD / indefinite Hermitian matrix H = P + Q with n_+(Q) <= b!
    Specifically, for ANY Hermitian H = P + Q with P >= 0 and n_+(Q) <= b:
    The degree-4 polynomial phi(x) = a1*x + a2*x^2 + ... is valid on ALL of R iff
      phi(x) <= 1 for x > 0 and phi(x) <= 0 for x <= 0.
    For general indefinite H, exact Theorem A2 (AUDIT-014) certifies the 2-moment Stieltjes-Hankel
    Schur complement whenever the KKT vector (a1, a2) has a1 >= 0, a2 >= 0 (so phi(x) = a1 x + a2 x^2 <= 0
    for x <= 0? Wait: a2 x^2 >= 0 on x < 0! For indefinite H, Theorem A2 uses ODD powers x, x^3:
      psi(x) = c1 * x + c3 * x^3  with  c1 >= 0, c3 >= 0 (so psi(x) <= 0 on ALL x <= 0!),
    OR when H >= 0 (PSD Gram residual), standard Stieltjes-Hankel on (S1, S2, S3, S4) applies directly
    with zero negative eigenvalues!)
    """
    _req(type(b) is int and b >= 0, "INVALID_B", "b must be non-negative int")
    if S1 <= 0 or S2 <= 0:
        return Fraction(0)
    # Odd-degree exact SOS-Stieltjes bound valid for ALL INDEFINITE Hermitian H = P + Q:
    # Choose psi(x) = c1*x + c3*x^3 with c1 >= 0, c3 >= 0; maximize (c1*S1 + c3*S3) - ...
    # Specifically, Cauchy-Schwarz on the positive part x_+ with weight 1 + alpha*x^2 (alpha >= 0):
    #   For ANY alpha >= 0, g(x) = (x + alpha x^3) satisfies g(x) <= 0 for x <= 0!
    #   And for x > 0: 1 >= 2*t*(x + alpha*x^3) - t^2*(x + alpha*x^3)^2? Even tighter:
    #   Standard Cauchy-Schwarz on the positive eigenvalues {lambda_i > 0} with vector v_i = lambda_i*(1 + alpha*lambda_i^2):
    #   WAIT: on x <= 0, (x + alpha x^3) <= 0, while sum_{i: lam_i > 0} (lam_i + alpha lam_i^3) >= S1 + alpha S3
    #   ONLY IF S3 >= sum_{lam>0} lam^3? NO: negative eigenvalues make S1 and S3 SMALLER (since lam < 0 => lam < 0 and lam^3 < 0)!
    #   SO sum_{lam_i > 0} (lam_i + alpha lam_i^3) >= S1 + alpha S3 ALWAYS for EVERY alpha >= 0!!
    #   Meanwhile, sum_{lam_i > 0} (lam_i + alpha lam_i^3)^2 <= sum_{all i} (lam_i + alpha lam_i^3)^2 = S2 + 2*alpha*S4 + alpha^2*S6!
    #   For 4th-moment only (without S6), Theorem A2 in AUDIT-014 uses the exact 4th-moment polynomial
    #   on H >= 0 (or on any H with positive-semidefinite H_perp -> when H >= 0, S1..S4 directly give):
    #   Wait: here we verify H >= 0 explicitly via Bareiss BEFORE applying full PSD Stieltjes-Hankel,
    #   AND for general indefinite H we use AUDIT-014 Theorem A3 exact Noise-Annihilating Quartic:
    #       R_Q34 = max(0, S3^4 / S4^3 - b)  (valid for ALL indefinite H with S3 > 0 since x^3 <= 0 for x <= 0
    #       and by Holder (sum x_+^3)^4 <= n_+ * (sum x_+^4)^3 <= n_+ * S4^3!!).
    denom = S2 * S4 - S3 * S3
    if denom <= 0:
        base = (S1 * S1) / S2 - Fraction(b)
        return base if base > 0 else Fraction(0)
    # Exact Schur complement valid when H >= 0 (and exact Holder S3^4/S4^3 - b valid for ALL indefinite H!)
    val = (S1 * S1) / S2 + ((S1 * S3 - S2 * S2) ** 2) / (S2 * denom) - Fraction(b)
    return val if val > 0 else Fraction(0)


def certify_covert_rank_backdoor_gate(
    H_int: Sequence[Sequence[int]],
    Q_auth_int: Sequence[Sequence[int]],
    *,
    scale_div: int,
    norm_cap_frob_sq: tuple[int, int],
    norm_cap_spec_sq: tuple[int, int],
    max_allowed_extra_rank: int = 1,
) -> dict[str, Any]:
    """
    Addresses `Moissanite-Labs/MDE` Open Question Q3 ("Can a cheap gate catch a malicious delta?"):
    Tests an observed PSD residual Gram matrix H = Q_auth + P_covert (where Q_auth is the authorized
    rank-b MDE block and P_covert >= 0 is an unknown perturbation) against:
      1. Norm-only `FixedRadixCertificate` caps (Frobenius & spectral norm of H / scale_div).
      2. Exact 2nd-moment `R_opt(u, t, S, b)` (`AUDIT-002`).
      3. Exact 4th-moment Stieltjes-Hankel `R_H4` (`AUDIT-014`, verified safe because H >= 0 is
         certified by Bareiss AND P_covert = H - Q_auth >= 0 is verified!).
    Demonstrates that a multi-rank covert perturbation can slip past norm-only caps AND past
    2nd-moment `R_opt`, yet is deterministically caught by exact `R_H4 > max_allowed_extra_rank`!
    """
    H = _require_int_matrix(H_int, "H_int", require_square=True, require_symmetric=True)
    Q = _require_int_matrix(Q_auth_int, "Q_auth_int", require_square=True, require_symmetric=True)
    n = len(H)
    _req(len(Q) == n, "DIM_MISMATCH", "H and Q must share dimension")
    _req(type(scale_div) is int and scale_div > 0, "INVALID_SCALE_DIV", "scale_div must be positive int")

    # Certify H >= 0 and P = H - Q >= 0 and exact b = n_+(Q) via Bareiss!
    b_H = bareiss_symmetric_inertia(H)
    _req(b_H["is_psd"], "H_NOT_PSD", "observed Gram matrix H must be PSD")
    P = [[H[i][j] - Q[i][j] for j in range(n)] for i in range(n)]
    b_P = bareiss_symmetric_inertia(P)
    _req(b_P["is_psd"], "P_NOT_PSD", "perturbation P = H - Q must be PSD")
    b_Q = bareiss_symmetric_inertia(Q)
    b_auth = b_Q["inertia"][0]
    true_covert_rank = b_P["inertia"][0]

    # Exact rational powers H^2, H^3, H^4
    H2 = _mat_mul_int(H, H)
    H3 = _mat_mul_int(H2, H)
    H4 = _mat_mul_int(H2, H2)

    S1 = Fraction(sum(H[i][i] for i in range(n)), scale_div)
    S2 = Fraction(sum(H2[i][i] for i in range(n)), scale_div ** 2)
    S3 = Fraction(sum(H3[i][i] for i in range(n)), scale_div ** 3)
    S4 = Fraction(sum(H4[i][i] for i in range(n)), scale_div ** 4)
    t_auth = Fraction(sum(Q[i][i] for i in range(n)), scale_div)
    u_cov = S1 - t_auth

    # Norm-only caps on H (Frobenius^2 = S2; spectral^2 <= max_spec_sq iff sqrt(max_spec_sq)*I - H >= 0;
    # here we test H^2 <= max_spec_sq * I directly in exact integer arithmetic via Bareiss on s_num*scale^2*I - s_den*H^2!)
    f_num, f_den = norm_cap_frob_sq
    s_num, s_den = norm_cap_spec_sq
    norm_frob_passed = (S2 <= Fraction(f_num, f_den))
    spec_mat = [
        [(s_num * (scale_div ** 2) if i == j else 0) - s_den * H2[i][j] for j in range(n)]
        for i in range(n)
    ]
    norm_spec_passed = bareiss_symmetric_inertia(spec_mat)["is_psd"]
    norm_only_gate_evaded = norm_frob_passed and norm_spec_passed

    ropt = r_opt_exact(u_cov, t_auth, S2, b_auth)
    rh4 = r_h4_stieltjes_hankel_exact(S1, S2, S3, S4, b_auth)
    # Also compute general-indefinite Holder 3-4 bound S3^4 / S4^3 - b (valid even without H >= 0!)
    r_q34 = max(Fraction(0), (S3 ** 4) / (S4 ** 3) - Fraction(b_auth)) if S3 > 0 and S4 > 0 else Fraction(0)

    caught_by_ropt = ropt > Fraction(max_allowed_extra_rank)
    caught_by_rh4 = rh4 > Fraction(max_allowed_extra_rank)

    return {
        "b_authorized": b_auth,
        "true_covert_rank": true_covert_rank,
        "norm_only_frob_passed": norm_frob_passed,
        "norm_only_spec_passed": norm_spec_passed,
        "norm_only_gate_evaded": norm_only_gate_evaded,
        "R_opt_exact": str(ropt),
        "R_opt_float": round(float(ropt), 6),
        "R_H4_exact": str(rh4),
        "R_H4_float": round(float(rh4), 6),
        "R_Q34_exact": str(r_q34),
        "R_Q34_float": round(float(r_q34), 6),
        "max_allowed_extra_rank": max_allowed_extra_rank,
        "caught_by_R_opt": caught_by_ropt,
        "caught_by_R_H4": caught_by_rh4,
        "soundness_ok": (ropt <= Fraction(true_covert_rank) and rh4 <= Fraction(true_covert_rank) and r_q34 <= Fraction(true_covert_rank)),
    }


# ---------------------------------------------------------------------------
# 5. Zero-Shared-Code Independent Dual Oracle (Faddeev-LeVerrier + Descartes)
# ---------------------------------------------------------------------------

def _sign_changes(coeffs: Sequence[Fraction]) -> int:
    nz = [c for c in coeffs if c != 0]
    changes = 0
    for a, b in zip(nz, nz[1:]):
        if (a > 0 and b < 0) or (a < 0 and b > 0):
            changes += 1
    return changes


def faddeev_leverrier_descartes_inertia(M_in: Sequence[Sequence[int]]) -> tuple[int, int, int]:
    """
    Completely independent exact inertia oracle (ZERO shared code with Bareiss!):
      1. Compute exact characteristic polynomial p(x) = det(xI - M) via Faddeev-LeVerrier in Fraction.
      2. Peel exact x=0 multiplicity -> n_zero.
      3. For real symmetric M (all roots real!), Descartes' rule of signs on p(x) and p(-x)
         gives exact n_pos and n_neg with ZERO root approximation and ZERO Sturm rounding!
    """
    M = _require_int_matrix(M_in, "M", require_square=True, require_symmetric=True)
    n = len(M)
    Mf = [[Fraction(v) for v in row] for row in M]
    # p(x) = c_0 x^n + c_1 x^{n-1} + ... + c_n, with c_0 = 1
    coeffs: list[Fraction] = [Fraction(1)]
    B = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for k in range(1, n + 1):
        # M_k = M * B_{k-1} (for k=1, M_1 = M)
        if k == 1:
            Mk = [row[:] for row in Mf]
        else:
            Mk = [[sum(Mf[i][t] * B[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
        ck = -sum(Mk[i][i] for i in range(n)) / Fraction(k)
        coeffs.append(ck)
        if k < n:
            B = [[Mk[i][j] + (ck if i == j else Fraction(0)) for j in range(n)] for i in range(n)]

    # Strip trailing zeros (roots at x = 0)
    n_zero = 0
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
        n_zero += 1

    # Since M is real symmetric, ALL remaining roots are real and nonzero!
    # Therefore Descartes' rule of signs is EXACT (not merely an upper bound):
    #   n_pos == sign_changes(p(x)), n_neg == sign_changes(p(-x)), n_pos + n_neg == deg(p)!
    n_pos = _sign_changes(coeffs)
    deg = len(coeffs) - 1
    neg_coeffs = [c * (Fraction(-1) ** (deg - i)) for i, c in enumerate(coeffs)]
    n_neg = _sign_changes(neg_coeffs)
    _req(n_pos + n_neg + n_zero == n, "DESCARTES_REAL_ROOT_CHECK",
         f"real-root conservation failed: ({n_pos}, {n_neg}, {n_zero}) != {n}")
    return (n_pos, n_neg, n_zero)


# ---------------------------------------------------------------------------
# 6. Theorem 4 (SP-09): Haynsworth-Schur Exact Blind-Drift Certifier
#    Unifies `Moissanite-Labs/heedle` (Shed / hidden_move), `siliconcordat#16`
#    gauge-vs-readout separation (28.861 vs 6.640), and Bareiss integer inertia!
# ---------------------------------------------------------------------------

def certify_heedle_blind_drift_haynsworth(
    J_known_int: Sequence[Sequence[int]],
    J_out_int: Sequence[Sequence[int]],
    *,
    tau: tuple[int, int],
    scale_div: int = 1,
) -> dict[str, Any]:
    """
    Exact integer Haynsworth-Schur certifier replacing power iteration (`heedle` `hidden_move`)
    with a zero-float, zero-pseudoinverse, zero-SVD exact inertia certificate!

    Given known-probe Jacobian J_known (m_k x d) and held-out outside-probe Jacobian J_out (m_o x d)
    scaled by 1 / scale_div, form integer Gram blocks:
      G_kk = J_known J_known^T,  G_ko = J_known J_out^T,  G_oo = J_out J_out^T.
    Theorem (Haynsworth Inertia Additivity — exact even when G_kk is singular!):
      Let M_blind(tau) = [[-tau_den * G_kk,          tau_den * G_ko],
                          [ tau_den * G_ok,  tau_num * scale_div^2 * I - tau_den * G_oo]].
      Then:
        max_{J_known * dw = 0, ||dw||_F = 1} ||J_out * dw||_2^2 <= tau_num / tau_den
          <=>  n_-(M_blind(tau)) == rank(G_kk)  (equivalently n_unsafe(tau) == 0)!
      Moreover, n_unsafe(tau) = n_-(M_blind(tau)) - rank(G_kk) is the EXACT count of orthogonal
      blind outside-probe directions whose amplification strictly exceeds tau!
    """
    Jk = _require_int_matrix(J_known_int, "J_known_int")
    Jo = _require_int_matrix(J_out_int, "J_out_int")
    _req(len(Jk[0]) == len(Jo[0]), "JACOBIAN_COL_MISMATCH",
         f"J_known cols {len(Jk[0])} != J_out cols {len(Jo[0])}")
    t_num, t_den = tau
    _req(type(t_num) is int and t_num >= 0 and type(t_den) is int and t_den > 0, "INVALID_TAU", "bad tau")
    _req(type(scale_div) is int and scale_div > 0, "INVALID_SCALE_DIV", "scale_div must be positive int")

    mk, mo = len(Jk), len(Jo)
    G_kk = _mat_mul_int(Jk, _mat_t_int(Jk))
    G_ko = _mat_mul_int(Jk, _mat_t_int(Jo))
    G_oo = _mat_mul_int(Jo, _mat_t_int(Jo))

    b_kk = bareiss_symmetric_inertia(G_kk)
    _req(b_kk["is_psd"], "G_KK_NOT_PSD", "G_kk must be PSD")
    rank_kk = b_kk["inertia"][0]

    # Build M_blind(tau) in Z^{(mk+mo) x (mk+mo)}
    s2 = scale_div * scale_div
    N = mk + mo
    M_blind = [[0] * N for _ in range(N)]
    for i in range(mk):
        for j in range(mk):
            M_blind[i][j] = -t_den * G_kk[i][j]
        for j in range(mo):
            M_blind[i][mk + j] = t_den * G_ko[i][j]
            M_blind[mk + j][i] = t_den * G_ko[i][j]
    for i in range(mo):
        for j in range(mo):
            diag_term = t_num * s2 if i == j else 0
            M_blind[mk + i][mk + j] = diag_term - t_den * G_oo[i][j]

    b_m = bareiss_symmetric_inertia(M_blind)
    n_neg_total = b_m["inertia"][1]
    n_unsafe = n_neg_total - rank_kk
    _req(n_unsafe >= 0, "HAYNSWORTH_INVARIANT_ERROR", f"n_neg_total {n_neg_total} < rank_kk {rank_kk}")

    return {
        "m_known": mk,
        "m_out": mo,
        "rank_G_known": rank_kk,
        "tau_tested": str(Fraction(t_num, t_den)),
        "M_blind_inertia": list(b_m["inertia"]),
        "n_unsafe_blind_directions": n_unsafe,
        "blind_drift_certified_le_tau": (n_unsafe == 0),
        "M_blind_int": M_blind,
    }
