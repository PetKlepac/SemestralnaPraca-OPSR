import numpy as np
from scipy.linalg import solve_continuous_are
from semestralka.system import A, B


# ====================== VÝPOČET K ======================
Q = np.diag([100.0, 0.01, 100.0, 0.01])
R = np.array([[1.0]])
P = solve_continuous_are(A, B, Q, R)
K = (np.linalg.inv(R) @ B.T @ P)   # tvar (1, 4)


# ====================== LQR RIADENIE ======================
def lqr_control(state, t):
    x, dx, theta, dtheta = state
    lin_state = np.array([x, dx, theta - np.pi, dtheta])
    u_lqr = - (K @ lin_state).item()
    return u_lqr