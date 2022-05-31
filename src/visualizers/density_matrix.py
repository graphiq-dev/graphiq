import matplotlib.pyplot as plt
import seaborn as sns
import qutip as qt
import numpy as np

cmap_uni = sns.color_palette("crest", as_cmap=True)
cmap_div = sns.diverging_palette(220, 20, as_cmap=True)


def density_matrix_heatmap(rho):
    """
    Plots a density matrix as 2D heatmap, one for the real components and one for the imaginary

    :param rho: a complex numpy array representing the density matrix
    :type rho: numpy.dnarray
    :return: fig (figure handle), axs (list of axes handles)
    :rtype: matplotlib.figure, matplotlib.axes
    """
    if type(rho) is qt.Qobj:
        rho = rho.full()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[10, 4])

    kwargs = dict(vmin=-1, vmax=1, cmap=cmap_div, linewidths=.5, square=True)
    axs[0].set(title="Real")
    axs[1].set(title="Imaginary")

    sns.heatmap(rho.real, ax=axs[0], **kwargs)
    sns.heatmap(rho.imag, ax=axs[1], **kwargs)

    return fig, axs


def density_matrix_bars(rho):
    """
    Plots a density matrix as 3D bar plots, one for the real components and one for the imaginary

    :param rho: a complex numpy array representing the density matrix
    :type rho: numpy.dnarray
    :return: fig (figure handle), axs (list of axes handles)
    :rtype: matplotlib.figure, matplotlib.axes
    """
    def bar_plot(deltaz, ax):
        n = deltaz.shape[0]
        X, Y = np.meshgrid(np.arange(n), np.arange(n))
        x, y = X.flatten() - 0.5, Y.flatten() - 0.5
        z = 1
        dx, dy = 0.8, 0.8
        deltaz = deltaz.flatten()
        max_height = 0.25  # get range of colorbars so we can normalize
        min_height = -0.25
        colors = cmap_div(deltaz.ravel() * 0.8, alpha=1 - deltaz.ravel())
        ax.bar3d(x, y, z, dx, dy, deltaz, color=colors)
        ax.set(zlim=[1+min_height, 2-max_height])

        ax.set(
            # xticks=[0, 1, 2, 3],
            # yticks=[0, 1, 2, 3],
            zticklabels=[]
        )
        return ax

    if type(rho) is qt.Qobj:
        rho = rho.full()

    fig = plt.figure()  # create a canvas, tell matplotlib it's 3d
    axs = [fig.add_subplot(1, 2, k, projection='3d') for k in range(1, 3)]

    dz = rho.real
    bar_plot(dz, ax=axs[0])
    axs[0].set(title="Real")

    dz = rho.imag
    bar_plot(dz, ax=axs[1])
    axs[1].set(title="Imaginary")

    return fig, axs
