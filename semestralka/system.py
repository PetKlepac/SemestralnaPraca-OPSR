import numpy as np
import sympy as sp
from control import ctrb


# ====================== DEFINICIA ULOHY ======================
M = 0.37     # hmotnosť vozíka [kg]
m = 0.127    # hmotnosť kyvadla [kg]
l = 0.1685   # dĺžka kyvadla k ťažisku [m]
b = 0.1      # koeficient tlmenia vozíka [Ns/m]
g = 9.81     # gravitačné zrýchlenie [m/s²]
I = 0.004812 # moment zotrvačnosti kyvadla [kg·m²]

# x = [x, dx, theta, dtheta]
# x - poloha vozíka
# dx - rýchlosť vozíka
# theta - uhol kyvadla
# dtheta - uhlová rýchlosť kyvadla
# vstup: u = F     (sila na vozík)


# ====================== NELINEARNY MODEL - funkcia f ======================
x, dx, theta, dtheta, F = sp.symbols('x dx theta dtheta F', real=True)
ddx, ddtheta = sp.symbols('ddx ddtheta', real=True)

eq1 = (M + m)*ddx + b*dx + m*l*ddtheta*sp.cos(theta) - m*l*dtheta**2*sp.sin(theta) - F
eq2 = (I + m*l**2)*ddtheta + m*g*l*sp.sin(theta) + m*l*ddx*sp.cos(theta)

sol = sp.solve([eq1, eq2], [ddx, ddtheta], dict=True)[0]
f = sp.Matrix([dx, sol[ddx], dtheta, sol[ddtheta]])


# ====================== LINEARIZÁCIA ======================
xp = {x: 0, dx: 0, theta: sp.pi, dtheta: 0}
A_sym = f.jacobian([x, dx, theta, dtheta]).subs(xp).doit().evalf()
B_sym = f.jacobian([F]).subs(xp).doit().evalf()
A = np.array(A_sym, dtype=float)
B = np.array(B_sym, dtype=float).reshape(4, 1)


# ==================== OVERENIE KONTROLOVATELNOSTI ====================
C = ctrb(A, B)
rank = np.linalg.matrix_rank(C)
n = A.shape[0]  # počet stavov systému
assert rank == n, "Systém nie je kontrolovateľný"


# ==================== DYNAMIKA SYSTÉMU ====================
def nonlinear_dynamics(state, u, torque_dist=0.0):

    x, dx, theta, dtheta = state

    s = np.sin(theta)
    c = np.cos(theta)

    # --- Pohyb vozíka ---
    num_ddx = (
        u
        - b * dx
        + (m**2 * g * l**2 * s * c) / (I + m * l**2)
        + m * l * dtheta**2 * s
    )

    den_ddx = (
        M + m
        - (m**2 * l**2 * c**2) / (I + m * l**2)
    )

    ddx = num_ddx / den_ddx

    # --- Pohyb kyvadla ---
    num_ddtheta = (
        -m * g * l * s
        - m * l * c * (
            u
            - b * dx
            + m * l * dtheta**2 * s
        ) / (M + m) + torque_dist
    )

    den_ddtheta = (
        I + m * l**2
        - (m**2 * l**2 * c**2) / (M + m)
    )

    ddtheta = num_ddtheta / den_ddtheta

    return np.array([
        dx,
        ddx,
        dtheta,
        ddtheta
    ], dtype=float)


# ==================== TVORBA PORUCHY ====================
def create_disturbance_array(t_grid, disturbances, noise_max=0.0, seed=None):

    d_array = np.zeros_like(t_grid, dtype=float)

    # === Deterministické poruchy (obdĺžnikové impulzy) ===
    for t_start, amp, dur in disturbances:
        mask = (t_grid >= t_start) & (t_grid < t_start + dur)
        d_array[mask] += amp

    # === White Noise (uniform distribution) ===
    if noise_max > 0.0:
        if seed is not None:
            np.random.seed(seed)  # pre reprodukovateľnosť

        noise = np.random.uniform(-noise_max, noise_max, size=len(t_grid))
        d_array += noise

    return d_array


