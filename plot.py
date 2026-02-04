import pickle
import numpy as np
import matplotlib.pyplot as plt


COLORS = {
    0.5: "tab:blue",
    1.0: "tab:green",
    2.0: "tab:red",
}


def mean_over_episodes(sequences):
    max_len = max(len(s) for s in sequences)
    mat = np.full((len(sequences), max_len), np.nan)

    for i, s in enumerate(sequences):
        mat[i, :len(s)] = s

    return np.nanmean(mat, axis=0)


def plot_paper_analysis():
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)

    plt.figure(figsize=(13, 10))

    plt.subplot(2, 2, 1)

    for mass, color in COLORS.items():
        height_seqs = results[mass]["height"]
        mean_height = mean_over_episodes(height_seqs)
        plt.plot(
            mean_height,
            color=color,
            linewidth=2,
            label=f"Масса *{mass}"
        )

    plt.axhline(0.28, linestyle="--", color="black", label="Ключевая высота")
    plt.axhline(0.15, linestyle="--", color="red", label="Граница провала эпизода")
    plt.ylabel("Высота (метров)")
    plt.title("Средняя высота тела")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 2)

    for mass, color in COLORS.items():
        roll_seqs = results[mass]["roll"]
        mean_roll = mean_over_episodes(roll_seqs)
        plt.plot(
            mean_roll,
            color=color,
            linewidth=2,
            label=f"Масса *{mass}"
        )

    plt.axhline(0, color="black", alpha=0.3)
    plt.ylabel("Крен (радиан)")
    plt.title("Средний угол наклона")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 3)

    for mass, color in COLORS.items():
        reward_seqs = results[mass]["reward_steps"]
        mean_reward = mean_over_episodes(reward_seqs)
        plt.plot(
            mean_reward,
            color=color,
            linewidth=2,
            label=f"Масса *{mass}"
        )

    plt.xlabel("Шаг")
    plt.ylabel("Награда")
    plt.title("Значение награды")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 4)

    for mass, color in COLORS.items():
        energy = results[mass]["energy"]
        plt.plot(
            range(len(energy)),
            energy,
            color=color,
            label=f"Масса *{mass}"
        )

    plt.xlabel("Эпизод")
    plt.ylabel("Суммарная энергия")
    plt.title("Суммарная энергия на эпизод запуска")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_paper_analysis()
