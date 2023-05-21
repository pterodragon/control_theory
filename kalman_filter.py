from typing import Optional
import numpy as np
from numpy.linalg import inv


class KalmanFilter(object):
    def __init__(
        self,
        F: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ):
        """
        F: state transition
        B: control input model
        H: observation model
        Q: process noise covariance matrix
        R: observation noise covariance matrix
        P: a priori estimate
        x0: initial state
        """

        if F is None or H is None:
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = F.shape[0]

        self.F = F
        self.H = H
        self.B = np.eye(self.n) if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u=None):
        u = np.zeros(self.n).T if u is None else u
        x = self.x
        P = self.P
        B = self.B
        Q = self.Q
        F = self.F

        x = (F @ x) + (B @ u)
        P = (F @ P @ F.T) + Q
        self.x = x
        self.P = P
        return self.x

    def update(self, z):
        x = self.x
        P = self.P
        H = self.H
        R = self.R
        I = np.eye(self.n)

        S = R + H @ P @ H.T
        K = P @ H.T @ inv(S)
        x = x + K @ (z - H @ x)
        P = (I - K @ H) @ P

        self.x = x
        self.P = P


def example():
    dt = 1.0 / 60
    # dynamics for the ODE y'' + y' = 0; y(0) = 0; y'(0) = 2; solution is y = 2 * sin(x)
    F = np.array([[1, dt, 0], [0, 1, dt], [0, -1, 0]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    n_samples = 200

    x = np.linspace(0, 10, n_samples)
    # measurements = -(x**2 + 2 * x - 2) + np.random.normal(0, 2, n_samples)
    ground_truths = 2 * np.sin(x)
    measurements = 2 * np.sin(x) + np.random.normal(0, 1, n_samples)
    x0 = np.array([0, 2, -2]).T

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0)
    predictions = []

    for z in measurements:
        predictions.append((H @ kf.predict())[0])
        kf.update(z)

    import matplotlib.pyplot as plt

    plt.plot(range(len(measurements)), measurements, label="Measurements")
    plt.plot(
        range(len(predictions)), np.array(predictions), label="Kalman Filter Prediction"
    )
    plt.plot(
        range(len(ground_truths)), np.array(ground_truths), label="Ground Truth"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    example()
