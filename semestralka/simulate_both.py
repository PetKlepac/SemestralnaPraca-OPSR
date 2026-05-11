import matplotlib.pyplot as plt

from semestralka.charts import plot_states_comparison, plot_control_comparison

import semestralka.lqr_simulation as lqr
import semestralka.mpc_simulation as mpc

print("Obidve simulácie dokončené.")

plot_states_comparison(lqr.t_grid,
                       lqr.x_poloha, mpc.x_poloha,
                       lqr.x_uhol,   mpc.x_uhol,
                       title='Porovnanie LQR vs MPC')

plot_control_comparison(lqr.t_grid,
                        lqr.u, mpc.u, lqr.d,          # predpoklad rovnakej poruchy
                        title='Porovnanie LQR vs MPC – riadiaca sila')

plt.show()