import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from semestralka.charts import create_charts
from semestralka.mpc import non_lin_model_MPC


# ====================== RIEŠENIE ====================
t_start = 0
t_final = 10
t_steps = 1000

x0 = [0.0, 0.0, np.pi + 0.0, 0.0]  # začiatok v hornej polohe + odchýlka

disturbances = [
    (1.0, 3.0, 0.5),   # začiatok 1.0s, sila 3N, trvanie 0.5s
    (6.0, -1.5, 0.2),
]

u_history = []

print("Spúšťam simuláciu...")

sol = solve_ivp(
    fun=lambda t, y:
    non_lin_model_MPC(t, y, disturbances=disturbances, u_history=u_history),
    t_span=(0, t_final),
    y0=x0,
    t_eval=np.linspace(t_start, t_final, t_steps),
    method='RK45',
    rtol=1e-6,
    atol=1e-8
)

print("Simulácia dokončená.")

u_array = np.array(u_history[:len(sol.t)])


# ====================== VIZUALIZÁCIA ======================
create_charts(sol, u_history=u_array, title='Simulácia nelineárneho modelu s MPC')
plt.show()
