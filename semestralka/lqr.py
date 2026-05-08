import numpy as np
from scipy.linalg import solve_continuous_are
from semestralka.system_and_linearization import M, m, l, b, g, I, A, B


# ====================== VÝPOČET K ======================
Q = np.diag([100.0, 0.01, 100.0, 0.01])
R = np.array([[1.0]])
P = solve_continuous_are(A, B, Q, R)
K = (np.linalg.inv(R) @ B.T @ P)   # tvar (1, 4)


# ==================== NELINEÁRNY MODEL POUZIVAJUCI LQR ====================
def non_lin_model_LQR(t, non_lin_state, u_min,      # ← now configurable
                      u_max, disturbances=None, u_history=None):
    x, dx, theta, dtheta = non_lin_state

    lin_state = np.array([x, dx, theta - np.pi, dtheta])

    u_lqr = - (K @ lin_state).item()  # control force from LQR

    # === External disturbance force ===
    F_dist = 0.0
    if disturbances:
        for t_start, F_value, duration in disturbances:  # ← now 3 values
            if t_start <= t < t_start + duration:
                F_dist += F_value  # can have multiple overlapping

    u = u_lqr + F_dist

    u = np.clip(u, u_min, u_max)

    if u_history is not None:
        u_history.append(u)

    s = np.sin(theta)
    c = np.cos(theta)

    num_ddx = (
        u - b * dx
        + (m ** 2 * g * l ** 2 * s * c) / (I + m * l ** 2)
        + m * l * dtheta ** 2 * s
    )

    den_ddx = M + m - (m ** 2 * l ** 2 * c ** 2) / (I + m * l ** 2)

    ddx = num_ddx / den_ddx

    num_ddtheta = (
        -m * g * l * s
        - m * l * c * (u - b * dx + m * l * dtheta ** 2 * s) / (M + m)
    )

    den_ddtheta = (I + m * l ** 2) - (m ** 2 * l ** 2 * c ** 2) / (M + m)

    ddtheta = num_ddtheta / den_ddtheta

    return np.array([float(dx), float(ddx), float(dtheta), float(ddtheta)], dtype=float)