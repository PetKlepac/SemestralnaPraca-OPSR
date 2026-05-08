import numpy as np
import cvxpy as cp
from scipy import linalg
from semestralka.system_and_linearization import M, m, l, b, g, I, A, B


# ====================== MPC PARAMETRE ======================
h = 0.02   # časový krok diskretizácie [s]
N = 50     # predikčný horizont
Q = np.diag([100.0, 0.01, 100.0, 0.01])
R = np.array([[1.0]])

u_min = -10.0
u_max =  10.0


# ====================== DISKRETIZÁCIA A TERMINÁLNA MATICA ======================
Ad = np.eye(4) + h * A
Bd = h * B
infP = linalg.solve_discrete_are(Ad, Bd, Q, R)


# ====================== MPC OPTIMALIZAČNÝ PROBLÉM ======================
x_mpc = cp.Variable((4, N + 1))
u_mpc = cp.Variable((1, N))
x_init = cp.Parameter(4)

constraints = [
    x_mpc[:, 1:] == Ad @ x_mpc[:, :-1] + Bd @ u_mpc,
    x_mpc[:, 0] == x_init,
    u_mpc >= u_min,
    u_mpc <= u_max,
]

LQ = np.linalg.cholesky(Q)
LR = np.linalg.cholesky(R)

objective = cp.Minimize(
    cp.sum_squares(LQ.T @ x_mpc[:, :-1]) +
    cp.sum_squares(LR.T @ u_mpc) +
    cp.quad_form(x_mpc[:, N], infP)
)

problem = cp.Problem(objective, constraints)


# ====================== MPC RIADENIE ======================
def mpc_control(state, t):

    x_init.value = np.asarray(state).flatten()
    problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    return float(u_mpc.value[0, 0])


# ====================== NELINEÁRNY MODEL S MPC ======================
def non_lin_model_MPC(t, non_lin_state, disturbances=None, u_history=None):

    x, dx, theta, dtheta = non_lin_state
    lin_state = np.array([x, dx, theta - np.pi, dtheta])

    # MPC
    u_mpc_val = mpc_control(lin_state, t)

    # Pridanie poruchy
    F_dist = 0.0
    if disturbances:
        for t_start, F_value, duration in disturbances:
            if t_start <= t < t_start + duration:
                F_dist += F_value

    u = u_mpc_val + F_dist

    if u_history is not None:
        u_history.append(u)

    # Nelineárne rovnice
    s = np.sin(theta)
    c = np.cos(theta)

    num_ddx = (u - b*dx + (m**2 * g * l**2 * s * c) / (I + m*l**2) + m*l*dtheta**2 * s)
    den_ddx = M + m - (m**2 * l**2 * c**2) / (I + m*l**2)
    ddx = num_ddx / den_ddx

    num_ddtheta = (-m*g*l*s - m*l*c*(u - b*dx + m*l*dtheta**2*s) / (M + m))
    den_ddtheta = (I + m*l**2) - (m**2 * l**2 * c**2) / (M + m)
    ddtheta = num_ddtheta / den_ddtheta

    return np.array([dx, ddx, dtheta, ddtheta], dtype=float)

