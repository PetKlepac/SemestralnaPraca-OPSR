from matplotlib import pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


def plot_position(t, x_poloha, u=None, d=None, title=None):

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(t, x_poloha, 'b', linewidth=2, label='x')
    ax1.set_xlabel('Čas [s]')
    ax1.set_ylabel('Poloha x [m]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)

    if u is not None and d is not None:
        total_force = u + d
        ax2 = ax1.twinx()
        ax2.plot(t, total_force, 'k--', linewidth=2, alpha=0.85, label='u + d')
        ax2.set_ylabel('Celková sila u + d [N]', color='k')
        ax2.tick_params(axis='y', labelcolor='k')

    ax1.set_title('Poloha vozíka')

    # Lepšia legenda
    if u is not None and d is not None:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_angle(t, x_uhol, u=None, d=None, title=None):

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(t, x_uhol, 'r', linewidth=2, label='θ')
    ax1.set_xlabel('Čas [s]')
    ax1.set_ylabel('Uhol θ [°]', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.grid(True)

    if u is not None and d is not None:
        total_force = u + d  # rovnaký výpočet
        ax2 = ax1.twinx()
        ax2.plot(t, total_force, 'k--', linewidth=2, alpha=0.85, label='u + d')
        ax2.set_ylabel('Celková sila u + d [N]', color='k')
        ax2.tick_params(axis='y', labelcolor='k')

    ax1.set_title('Uhol kyvadla')

    if u is not None and d is not None:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig

def plot_control(t, u, d=None, title=None):

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(t, u, 'g', linewidth=2, label='u – riadiaca sila')

    if d is not None:
        ax.plot(t, d, color='orange', linestyle='--', linewidth=1.8, label='d – vonkajšia porucha')

    # === AUTOMATICKÉ NASTAVENIE OSY Y ===
    all_values = u
    if d is not None:
        all_values = np.concatenate([all_values, d])

    ymin = np.min(all_values) * 1.1
    ymax = np.max(all_values) * 1.1
    abs_max = max(abs(ymin), abs(ymax))
    ax.set_ylim([-abs_max, abs_max])

    ax.set_xlabel('Čas [s]')
    ax.set_ylabel('Sila [N]')
    ax.set_title('Riadiaca sila a porucha')
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_states_comparison(t, x_poloha_lqr, x_poloha_mpc,
                           x_uhol_lqr, x_uhol_mpc, title=None):

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Poloha vozíka
    axs[0].plot(t, x_poloha_lqr, 'b', linewidth=2, label='LQR')
    axs[0].plot(t, x_poloha_mpc, 'r', linewidth=2, label='MPC')
    axs[0].set_xlabel('Čas [s]')
    axs[0].set_ylabel('x [m]')
    axs[0].set_title('Poloha vozíka')
    axs[0].grid(True)
    axs[0].legend()

    # Uhol kyvadla
    axs[1].plot(t, x_uhol_lqr, 'b', linewidth=2, label='LQR')
    axs[1].plot(t, x_uhol_mpc, 'r', linewidth=2, label='MPC')
    axs[1].set_xlabel('Čas [s]')
    axs[1].set_ylabel('θ [°]')
    axs[1].set_title('Uhol kyvadla')
    axs[1].grid(True)
    axs[1].legend()

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    return fig


def plot_control_comparison(t, u_lqr, u_mpc, d=None, title=None):

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, u_lqr, 'b', linewidth=2, label='LQR – riadiaca sila')
    ax.plot(t, u_mpc, 'r', linewidth=2, label='MPC – riadiaca sila')

    if d is not None:
        ax.plot(t, d, 'k--', linewidth=1.5, alpha=0.7, label='d – porucha')

    values = [u_lqr, u_mpc]
    if d is not None:
        values.append(d)

    all_data = np.concatenate(values)
    margin = 0.15 * (np.max(all_data) - np.min(all_data))
    ymin = np.min(all_data) - margin
    ymax = np.max(all_data) + margin

    abs_max = max(abs(ymin), abs(ymax))
    ax.set_ylim([-abs_max, abs_max])

    ax.set_xlabel('Čas [s]')
    ax.set_ylabel('Sila [N]')
    ax.set_title('Porovnanie riadiacej sily LQR vs MPC')
    ax.grid(True)
    ax.legend(loc='upper right')

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    return fig