import time

import numpy as np
import plotly.graph_objects as go
from IPython.display import clear_output
from scipy.spatial import KDTree


def plot_graph(target, pc, title=''):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=target[:, 0], y=target[:, 1], z=target[:, 2],
                               mode='markers', marker=dict(size=4), name='target'))
    fig.add_trace(go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
                               mode='markers', marker=dict(size=4), name='pc'))
    fig.update_layout(title=title, scene=dict(aspectmode='data'))
    fig.show()


class ICP:
    """Point-to-point ICP: estimates R, t such that R @ source + t ≈ target."""

    def __init__(self, max_iterations=50, tolerance=1e-6, max_corr_dist=None,
                 visualize=False, plot_delay=0.1):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_corr_dist = max_corr_dist  # reject pairs farther apart than this (None = keep all)
        self.visualize = visualize
        self.plot_delay = plot_delay

        self.target = None
        self.tree = None

        # results of the last align()
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.aligned = None
        self.errors = []

    def set_target(self, target):
        self.target = target
        self.tree = KDTree(target)

    @staticmethod
    def best_fit_transform(A, B):
        mu_a = np.mean(A, axis=0)
        mu_b = np.mean(B, axis=0)

        A_c = A - mu_a
        B_c = B - mu_b

        H = A_c.T @ B_c
        U, S, Vt = np.linalg.svd(H)

        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = mu_b - R @ mu_a
        return R, t

    def _match(self, pc):
        distances, idx = self.tree.query(pc, workers=-1)
        src, matched = pc, self.target[idx]

        if self.max_corr_dist is not None:
            mask = distances < self.max_corr_dist
            src, matched, distances = src[mask], matched[mask], distances[mask]

        return src, matched, distances

    def align(self, source, R_init=None, t_init=None):
        if self.tree is None:
            raise RuntimeError('call set_target() before align()')

        R_total = np.eye(3) if R_init is None else R_init.copy()
        t_total = np.zeros(3) if t_init is None else t_init.copy()
        pc = source @ R_total.T + t_total

        self.errors = []
        prev_error = np.inf

        for i in range(self.max_iterations):
            src, matched, distances = self._match(pc)
            if len(src) < 3:
                raise RuntimeError(f'only {len(src)} matches left; max_corr_dist too small?')

            R_step, t_step = self.best_fit_transform(src, matched)

            pc = pc @ R_step.T + t_step

            t_total = R_step @ t_total + t_step
            R_total = R_step @ R_total

            error = np.sqrt(np.mean(distances ** 2))  # RMS of this iteration's matches
            self.errors.append(error)

            if self.visualize:
                self._plot(pc, i, error)

            if abs(prev_error - error) < self.tolerance:
                break
            prev_error = error

        self.R, self.t, self.aligned = R_total, t_total, pc
        return self.R, self.t

    def _plot(self, pc, i, error):
        clear_output(wait=True)
        plot_graph(self.target, pc, f'iteration {i}, RMS error {error:.4f}')
        time.sleep(self.plot_delay)
