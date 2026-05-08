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
# poruchy: počiatočná odchýlka uhla, vonkajšia sila na vozík


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

