from matplotlib import pyplot as plt
import numpy as np


def create_charts(sol, u_history, title=None):

    u_array = np.asarray(u_history)

    # ==================== FIGURE 1: States ====================
    fig1, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Poloha vozíka
    axs[0].plot(sol.t, sol.y[0], 'b', linewidth=2, label='x – poloha vozíka')
    axs[0].set_xlabel('Čas [s]')
    axs[0].set_ylabel('x [m]')
    axs[0].set_title('Poloha vozíka')
    axs[0].grid(True)

    # Uhol kyvadla
    axs[1].plot(sol.t, sol.y[2] * 180 / np.pi, 'r', linewidth=2, label='θ')
    axs[1].set_xlabel('Čas [s]')
    axs[1].set_ylabel('θ [°]')
    axs[1].set_title('Uhol kyvadla')
    axs[1].grid(True)

    fig1.suptitle(title, fontsize=14)
    fig1.tight_layout()

    # ==================== FIGURE 2: Control Input u ====================
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.plot(sol.t, u_array, 'g', linewidth=2, label='u')
    ax2.set_xlabel('Čas [s]')
    ax2.set_ylabel('u [N]')
    ax2.set_title('Riadiaca sila')
    ax2.grid(True)
    fig2.suptitle(title, fontsize=14)
    fig2.tight_layout()

    return fig1, fig2