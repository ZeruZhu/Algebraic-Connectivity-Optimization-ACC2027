from __future__ import annotations

import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import ArpackError, LinearOperator, eigsh


class CyclicDeleteFFTEvaluator:
    """Experimental matrix-free Laplacian solver for vertex-deleted circulants.

    Deletion destroys direct Fourier diagonalization, but padded convolution
    still applies the induced adjacency exactly. Degrees must be recomputed;
    a principal submatrix of the parent Laplacian would be incorrect.
    """

    def __init__(self, *, order: int, delete_nodes: tuple[int, ...]) -> None:
        self.n = int(order)
        deleted = np.asarray(sorted(set(delete_nodes)), dtype=np.int32)
        if self.n < 2 or np.any(deleted < 0) or np.any(deleted >= self.n):
            raise ValueError("Invalid cyclic order or deleted vertices.")
        self.keep = np.setdiff1d(np.arange(self.n), deleted)
        if self.keep.size < 2:
            raise ValueError("Need at least two retained vertices.")
        self.deleted = deleted
        self._lost_indices = (self.keep[:, None] - deleted[None, :]) % self.n
        self._inverse = (-np.arange(self.n)) % self.n
        self._start = np.random.default_rng(2701).normal(size=self.keep.size)
        self._start -= self._start.mean()
        self._start /= np.linalg.norm(self._start)
        self.fallback_count = 0
        self.disconnected_shortcuts = 0
        self.matvec_count = 0
        self.max_residual = 0.0

    def _operator(self, indicator: np.ndarray):
        degree = indicator.sum() - indicator[self._lost_indices].sum(axis=1)
        spectrum = np.fft.rfft(indicator)
        size = self.keep.size
        # The largest eigenvalue of shift*(I-11^T/size)-L is shift-lambda_2.
        # Unlike targeting a near-zero eigenvalue, its relative tolerance is well scaled.
        shift = 2.0 * float(degree.max()) + 1.0

        def matvec(vector):
            self.matvec_count += 1
            vector = np.asarray(vector).reshape(size)
            padded = np.zeros(self.n, dtype=np.float64)
            padded[self.keep] = vector
            adjacency_product = np.fft.irfft(spectrum * np.fft.rfft(padded), n=self.n)
            return (shift - degree) * vector + adjacency_product[self.keep] - shift * vector.mean()

        return LinearOperator((size, size), matvec=matvec, dtype=np.float64), degree

    def _dense_score(self, indicator: np.ndarray, degree: np.ndarray) -> float:
        indices = (self.keep[:, None] - self.keep[None, :]) % self.n
        laplacian = -indicator[indices]
        np.fill_diagonal(laplacian, degree)
        return float(eigvalsh(laplacian, subset_by_index=(1, 1), driver="evr", check_finite=False)[0])

    def lambda2_edges_from_index_matrix(self, generator_index_matrix):
        indices = np.asarray(generator_index_matrix, dtype=np.int32)
        if indices.ndim != 2:
            raise ValueError("generator_index_matrix must have shape [batch, degree].")
        batch = indices.shape[0]
        if batch == 0 or indices.shape[1] == 0:
            return np.zeros(batch), np.zeros(batch, dtype=np.int64)
        if np.any(indices <= 0) or np.any(indices >= self.n):
            raise ValueError("Generators must be nonidentity group elements.")
        indicators = np.zeros((batch, self.n), dtype=np.float64)
        indicators[np.arange(batch)[:, None], indices] = 1.0
        if not np.array_equal(indicators, indicators[:, self._inverse]):
            raise ValueError("Cyclic generator sets must be inverse-closed.")
        values = np.empty(batch)
        edges = np.empty(batch, dtype=np.int64)
        for row, indicator in enumerate(indicators):
            operator, degree = self._operator(indicator)
            edges[row] = int(round(float(degree.sum()) / 2.0))
            if not edges[row]:
                values[row] = 0.0
                continue
            divisor = int(np.gcd.reduce(np.flatnonzero(indicator), initial=self.n))
            if divisor > 1 and np.any(self.keep % divisor != self.keep[0] % divisor):
                self.disconnected_shortcuts += 1
                values[row] = 0.0
                continue
            try:
                eigenvalues, vectors = eigsh(
                    operator, k=1, which="LA", v0=self._start.copy(),
                    tol=1e-12, ncv=min(32, self.keep.size),
                    maxiter=300,
                )
                root = float(eigenvalues[0])
                value = 2.0 * float(degree.max()) + 1.0 - root
                residual = float(np.linalg.norm(operator @ vectors[:, 0] - root * vectors[:, 0]))
                self.max_residual = max(self.max_residual, residual)
                if not np.isfinite(value + residual) or residual > 1e-9 or value < -1e-9:
                    raise ArithmeticError("Unreliable iterative eigenpair.")
            except (ArpackError, ArithmeticError, np.linalg.LinAlgError):
                self.fallback_count += 1
                value = self._dense_score(indicator, degree)
            values[row] = max(value, 0.0)
        return values, edges
