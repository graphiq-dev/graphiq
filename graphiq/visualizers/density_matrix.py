import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

cmap_uni = sns.color_palette("crest", as_cmap=True)
cmap_div = sns.diverging_palette(220, 20, as_cmap=True)


def density_matrix_heatmap(rho, axs=None):
    """
    Plots a density matrix as 2D heatmap, one for the real components and one for the imaginary

    :param rho: a complex numpy array representing the density matrix
    :type rho: numpy.ndarray
    :param axs: axis to plot on
    :type axs: plt.Axis
    :return: fig (figure handle), axs (list of axes handles)
    :rtype: matplotlib.Figure, matplotlib.Axes
    """
    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[10, 4])
    else:
        fig = None

    kwargs = dict(vmin=-1, vmax=1, cmap=cmap_div, linewidths=0.5, square=True)
    axs[0].set(title="Real")
    axs[1].set(title="Imaginary")

    sns.heatmap(rho.real, ax=axs[0], **kwargs)
    sns.heatmap(rho.imag, ax=axs[1], **kwargs)

    return fig, axs


def density_matrix_bars(rho):
    """
    Plots a density matrix as 3D bar plots, one for the real components and one for the imaginary

    :param rho: a complex numpy array representing the density matrix
    :type rho: numpy.ndarray
    :return: fig (figure handle), axs (list of axes handles)
    :rtype: matplotlib.Figure, matplotlib.Axes
    """

    def bar_plot(deltaz, ax):
        if np.max(np.abs(deltaz)) < 0.25:
            max_height, min_height = (
                0.25,
                -0.25,
            )  # get range of colorbars so we can normalize
        elif np.max(np.abs(deltaz)) < 0.5:
            max_height, min_height = (
                0.5,
                -0.5,
            )  # get range of colorbars so we can normalize
        else:
            max_height, min_height = (
                1.0,
                -1.0,
            )  # get range of colorbars so we can normalize
        n = deltaz.shape[0]
        n_qubits = int(np.log2(n))
        X, Y = np.meshgrid(np.arange(n), np.arange(n))
        x, y = X.flatten() - 0.5, Y.flatten() - 0.5
        z = 0
        zlim = [min_height, max_height]
        dx, dy = 0.8, 0.8
        deltaz = deltaz.flatten()

        colors = cmap_div(
            deltaz.ravel() * 0.8, alpha=1 - 0.5 * (np.abs(deltaz.ravel()) / max_height)
        )
        ax.bar3d(x, y, z, dx, dy, deltaz, color=colors)
        ax.set(zlim=zlim)

        ticks = [i for i in range(n)]
        labels = ["" for tick in ticks]
        if n_qubits < 3:
            labels[0] = (
                r"$\vert" + "".join(["0" for i in range(n_qubits)]) + r"\rangle$"
            )
            labels[-1] = (
                r"$\vert" + "".join(["1" for i in range(n_qubits)]) + r"\rangle$"
            )
        else:
            labels[0] = r"$\vert 00...0\rangle$"
            labels[-1] = r"$\vert 1...1\rangle$"
        ax.set(
            xticks=ticks,  # todo, replace these with basis vectors, i.e., |00...0>, |00...1>, ...
            xticklabels=labels,
            yticks=ticks,
            yticklabels=labels,
            zticks=[min_height, max_height],
        )
        ax.set_xticklabels(labels, ha="right", va="center")
        ax.set_yticklabels(labels, ha="left", va="center")
        return ax

    fig = plt.figure()  # create a canvas, tell matplotlib it's 3d
    axs = [fig.add_subplot(1, 2, k, projection="3d") for k in range(1, 3)]

    dz = rho.real
    bar_plot(dz, ax=axs[0])
    axs[0].set(title="Real")

    dz = rho.imag
    bar_plot(dz, ax=axs[1])
    axs[1].set(title="Imaginary")

    return fig, axs
