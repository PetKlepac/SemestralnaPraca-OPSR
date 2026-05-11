import numpy as np
import cvxpy as cp
from scipy import linalg
from semestralka.system import A, B


# ====================== MPC PARAMETRE ======================
h = 0.02   # časový krok diskretizácie [s]
N = 50     # predikčný horizont
Q = np.diag([140.0, 0.01, 100.0, 0.01])
R = np.array([[1.0]])

u_min = -5.0
u_max =  5.0


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
    x, dx, theta, dtheta = state
    lin_state = np.array([x, dx, theta - np.pi, dtheta])
    x_init.value = np.asarray(lin_state).flatten()
    problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    return float(u_mpc.value[0, 0])