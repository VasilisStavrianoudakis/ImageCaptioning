# import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# from sklearn.metrics import confusion_matrix


def get_colors(num_colors: int):
    if num_colors < 5:
        colors = list(mcolors.BASE_COLORS)
        colors = [col for col in colors if "w" not in col and col not in ["r", "b"]]
    else:
        colors = list(mcolors.CSS4_COLORS)
        colors = [
            col
            for col in colors
            if "light" not in col and col not in ["white", "snow", "red", "blue"]
        ]
    colors = np.random.choice(colors, size=num_colors, replace=False)
    return colors


def plot_info(
    training_val,
    validation_val,
    y_name,
    saveto: str,
    extra_epochs: int = -1,
    events: Dict[str, int] = {},
):
    plt.figure()
    if events:
        colors = get_colors(len(events))
        for i, (event, x) in enumerate(events.items()):
            plt.axvline(x=x, color=colors[i], label=event, linestyle="--")

    epochs = (
        len(training_val) - extra_epochs if extra_epochs != -1 else len(training_val)
    )
    n_bins = epochs
    if extra_epochs != -1:
        if events["best_model"] + extra_epochs > epochs:
            n_bins = events["best_model"] + extra_epochs
        x_axis = list(range(events["best_model"] + 1))
        plt.plot(
            x_axis, training_val[: events["best_model"] + 1], "r-", label="training"
        )
        plt.plot(
            x_axis, validation_val[: events["best_model"] + 1], "b-", label="validation"
        )

        x_axis = list(
            range(events["best_model"], events["best_model"] + extra_epochs + 1)
        )
        extra_epochs_vals = training_val[epochs:]
        extra_epochs_vals.insert(0, training_val[events["best_model"]])
        plt.plot(x_axis, extra_epochs_vals, "r-")

        x_axis = list(range(events["best_model"], epochs))
        plt.plot(x_axis, training_val[events["best_model"] : epochs], "r--")
        plt.plot(x_axis, validation_val[events["best_model"] : epochs], "b--")
    else:
        x_axis = list(range(epochs))
        plt.plot(x_axis, training_val, "r-", label="training")
        x_axis = list(range(epochs))
        # plt.plot(x_axis, training_val, "r--", label="training")
        plt.plot(x_axis, validation_val, "b-", label="validation")

    # x_axis = list(range(epochs))
    # # plt.plot(x_axis, training_val, "r--", label="training")
    # plt.plot(x_axis, validation_val, "b-", label="validation")
    # plt.xticks(x_axis)
    # plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=n_bins, integer=True))
    # plt.gca().xaxis.set_major_locator(FixedLocator(list(range(n_bins))))
    if n_bins > 50:
        ticks = [num for num in range(n_bins) if num % 5 == 0]
    else:
        ticks = [num for num in range(n_bins) if num % 2 == 0]
    plt.xticks(ticks)
    plt.xticks(ticks, rotation=45, ha="right")

    plt.title(y_name)
    plt.ylabel(y_name)
    plt.xlabel("epoch")
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig(saveto, bbox_inches="tight")


def plot_hist(data: List[Tuple[str, List[int]]], saveto: str):
    plt.figure()
    # x = np.array(data[0][1])
    # q25 = np.quantile(x, 0.25)
    # q75 = np.quantile(x, 0.75)
    # bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    # bins = int(
    #     round((x.max() - x.min()) / bin_width)
    # )  # for some reason my numpy version messes up the round function and it returns a np.float.
    bins = np.arange(max(data[0][1]) + 1) - 0.5
    # bins = 3

    if len(data) < 4:
        colors = ["b", "g", "r"]
    else:
        colors = get_colors(len(data))

    fig, axs = plt.subplots(len(data))
    # fig.suptitle('Vertically stacked subplots')
    for idx, (label, d) in enumerate(data):
        # alpha = 0.9 if "pred" in label.lower() else 0.3
        # plt.hist(d, bins, alpha=alpha, label=label)
        axs[idx].hist(d, bins, label=label, color=colors[idx])
        axs[idx].legend(loc="best")
        # axs[idx].ylabel("Counts")
        # axs[idx].xlabel("Bins")

    # fig.legend(loc="best")
    fig.suptitle("Y Distribution")
    # fig.ylabel("Counts")
    # fig.xlabel("Bins")
    plt.savefig(saveto, bbox_inches="tight")
