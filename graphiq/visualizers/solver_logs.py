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


def plot_solver_logs(logs: dict):
    """
    Plots the cost value and circuit depth (mean and range), for both the population and hof, as a function of the
    solver iteration.

    :param logs: dictionary containing a log (pd.DataFrame) for population and hof
    :return:
    """
    fig, axs = plt.subplots(nrows=2, ncols=2, sharey="row", sharex="col")
    colors = ["teal", "orange"]
    for col, (log_name, log) in enumerate(logs.items()):
        c = colors[col]
        axs[0, col].plot(log["iteration"], log["cost_mean"], color=c, label=f"mean")
        axs[0, col].fill_between(
            log["iteration"],
            log["cost_min"],
            log["cost_max"],
            color=c,
            alpha=0.3,
            label=f"range",
        )

        axs[1, col].plot(log["iteration"], log["depth_mean"], color=c)
        axs[1, col].fill_between(
            log["iteration"], log["depth_min"], log["depth_max"], color=c, alpha=0.3
        )

        axs[0, col].set(title=f"{log_name}")

    axs[0, 0].legend()
    axs[0, 1].legend()

    axs[0, 0].set(ylabel="Cost value")
    axs[1, 0].set(ylabel="Circuit depth")
    axs[1, 0].set(xlabel="Iteration")
    axs[1, 1].set(xlabel="Iteration")

    return fig, axs
