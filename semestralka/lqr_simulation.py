import numpy as np
import matplotlib.pyplot as plt
from semestralka.charts import plot_position, plot_angle, plot_control
from semestralka.lqr import lqr_control
from semestralka.system import nonlinear_dynamics
from semestralka.system import create_disturbance_array


# ====================== PARAMETRE A PREMENNÉ ====================
t_start = 0
t_final = 10
dt = 0.02
t_grid = np.arange(t_start, t_final + dt, dt)

x0 = [0.0, 0.0, np.pi + 0.3, 0.0]  # začiatok v hornej polohe + odchýlka
x = x0.copy()                      # počiatočný krok
x_history = []
u_history = []

u_min = -5.0
u_max = 5.0


# ====================== NASTAVENIE PORÚCH ====================
disturbances = [
    (2.0, 2.5, 3.0),   # začiatok 1.0s, sila 2.5N, trvanie 3.0s
    (7.0, -1.0, 0.5),
]
d_array = create_disturbance_array(t_grid, disturbances, noise_max=0.0, seed=67)


# ====================== SIMULÁCIA ====================
print("Spúšťam simuláciu...")

for i, t in enumerate(t_grid):

    # ===== MPC RIADENIE =====
    u_mpc = lqr_control(x, t)

    u_mpc = np.clip(u_mpc, u_min, u_max)

    # ===== PRIDANIE PORUCHY =====
    u_total = u_mpc + d_array[i]

    # ===== ULOŽENIE DÁT =====
    x_history.append(x.copy())
    u_history.append(u_mpc)

    # ===== DYNAMIKA SYSTÉMU =====
    dxdt = nonlinear_dynamics(x, u_total)

    # ===== INTEGRÁCIA POMOCOU EULERA =====
    x = x + dt * dxdt

print("Simulácia dokončená.")


# ====================== TVORBA GRAFOV ====================
x_array = np.array(x_history)
x_poloha = x_array[:, 0]
x_uhol = x_array[:, 2] * 180 / np.pi

u = np.array(u_history)
d = d_array


# ====================== GRAFY ======================
if __name__ == "__main__":
    plot_angle(t_grid, x_uhol, u=u, d=d)
    plot_position(t_grid, x_poloha, u=u, d=d)
    plot_control(t_grid, u, d)
    plt.show()
else:
    pass