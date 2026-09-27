# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import Axes3D

from graphiq.visualizers.density_matrix import (
    cmap_div,
    density_matrix_bars,
    density_matrix_heatmap,
)

# Row-major entries 0.4, -0.1, -0.4, 0. The largest magnitude is 0.4, so the real
# panel's range is [-0.5, 0.5] and entry x sits at 0.5 + x on the colour map.
# In floats 0.5 + 0.5 * -0.4 / 0.5 is 0.09999999999999998, not 0.1; the exact
# equalities below hold because the colour map is a 256-entry lookup table and
# both land on entry 25.
RHO = np.array([[0.4, -0.1], [-0.4, 0.0]], dtype=complex)


@pytest.fixture
def bar_colours(monkeypatch):
    """The color= array each bar3d call receives, one entry per panel."""
    recorded = []
    bar3d = Axes3D.bar3d

    def spy(self, *args, color=None, **kwargs):
        recorded.append(np.asarray(color))
        return bar3d(self, *args, color=color, **kwargs)

    monkeypatch.setattr(Axes3D, "bar3d", spy)
    fig, _ = density_matrix_bars(RHO)
    plt.close(fig)
    return recorded


def rgb(position):
    return np.asarray(cmap_div(position))[:3]


def test_bar_colour_of_a_zero_entry_is_the_neutral_centre(bar_colours):
    real = bar_colours[0]
    np.testing.assert_array_equal(real[3, :3], rgb(0.5))


def test_negative_entries_of_different_size_get_different_colours(bar_colours):
    real = bar_colours[0]
    np.testing.assert_array_equal(real[1, :3], rgb(0.4))  # -0.1
    np.testing.assert_array_equal(real[2, :3], rgb(0.1))  # -0.4
    assert not np.array_equal(real[1, :3], real[2, :3])


def test_bar_colours_are_symmetric_about_zero(bar_colours):
    real = bar_colours[0]
    np.testing.assert_array_equal(real[0, :3], rgb(0.9))  # +0.4
    np.testing.assert_array_equal(real[2, :3], rgb(0.1))  # -0.4


def test_an_all_zero_panel_is_drawn_in_the_neutral_colour(bar_colours):
    # the imaginary part of a real matrix: every bar is 0
    imag = bar_colours[1]
    for colour in imag:
        np.testing.assert_array_equal(colour[:3], rgb(0.5))


def test_bar_alpha_is_unchanged(bar_colours):
    real, imag = bar_colours
    np.testing.assert_allclose(
        real[:, 3], 1 - 0.5 * np.abs([0.4, -0.1, -0.4, 0.0]) / 0.5
    )
    np.testing.assert_array_equal(imag[:, 3], np.ones(4))


def test_heatmap_returns_the_figure_of_the_axes_it_is_given():
    fig, axs = plt.subplots(nrows=1, ncols=2)
    try:
        returned_fig, returned_axs = density_matrix_heatmap(RHO, axs=axs)
        assert returned_fig is fig
        assert returned_axs is axs
    finally:
        plt.close(fig)


def test_heatmap_given_a_tuple_of_axes_returns_their_figure_and_the_tuple():
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs = tuple(axs)
    try:
        returned_fig, returned_axs = density_matrix_heatmap(RHO, axs=axs)
        assert returned_fig is fig
        assert returned_axs is axs
    finally:
        plt.close(fig)


def test_heatmap_given_axes_in_a_subfigure_returns_the_root_figure():
    fig = plt.figure()
    axs = fig.subfigures(1, 1).subplots(1, 2)
    try:
        returned_fig, _ = density_matrix_heatmap(RHO, axs=axs)
        assert returned_fig is fig
    finally:
        plt.close(fig)


def test_heatmap_given_axes_from_two_figures_returns_the_first_axes_figure():
    fig_real, ax_real = plt.subplots()
    fig_imag, ax_imag = plt.subplots()
    try:
        returned_fig, _ = density_matrix_heatmap(RHO, axs=[ax_real, ax_imag])
        assert returned_fig is fig_real
        # the imaginary heatmap is still drawn there
        assert len(ax_imag.collections) == 1
    finally:
        plt.close(fig_real)
        plt.close(fig_imag)


def test_heatmap_without_axes_still_opens_a_new_two_panel_figure():
    before = set(plt.get_fignums())
    fig, axs = density_matrix_heatmap(RHO)
    try:
        assert fig.number not in before
        assert [ax.figure for ax in axs] == [fig, fig]
    finally:
        plt.close(fig)
