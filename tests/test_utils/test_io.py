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
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from graphiq.io import IO


def test_io_init_no_path_uses_default_path():
    io = IO(verbose=False)
    assert io.path == IO.default_path


def test_io_init_str_path_is_converted_to_path(tmp_path):
    io = IO(path=str(tmp_path), verbose=False)
    assert io.path == tmp_path


def test_new_directory_does_not_touch_filesystem(tmp_path):
    io = IO.new_directory(path=tmp_path, folder="my_folder", verbose=False)
    assert not io.path.exists()


def test_new_directory_warns_only_when_folder_is_falsy(tmp_path, recwarn):
    IO.new_directory(path=tmp_path, folder="my_folder", verbose=False)
    assert len(recwarn.list) == 0

    IO.new_directory(path=tmp_path, folder="", verbose=False)
    assert len(recwarn.list) == 1
    assert "No folder entered" in str(recwarn.list[0].message)


def test_save_and_load_json_round_trip(tmp_path):
    io = IO(path=str(tmp_path), verbose=False)
    data = {"a": 1, "b": [1, 2, 3]}
    io.save_json(data, "data.json")
    assert io.load_json("data.json") == data


def test_save_and_load_txt_round_trip(tmp_path):
    io = IO(path=str(tmp_path), verbose=False)
    io.save_txt("hello world", "data.txt")
    assert io.load_txt("data.txt") == "hello world"


def test_save_and_load_dataframe_round_trip(tmp_path):
    io = IO(path=str(tmp_path), verbose=False)
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    io.save_dataframe(df, "data.csv")
    loaded = io.load_dataframe("data.csv")
    pd.testing.assert_frame_equal(loaded, df)


def test_save_and_load_np_array_round_trip(tmp_path):
    io = IO(path=str(tmp_path), verbose=False)
    arr = np.array([1.0, 2.0, 3.0])
    io.save_np_array(arr, "data.txt")
    loaded = io.load_np_array("data.txt")
    assert np.allclose(loaded, arr)


def test_save_figure_writes_file(tmp_path):
    io = IO(path=str(tmp_path), verbose=False)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    io.save_figure(fig, "figure.png")
    assert (tmp_path / "figure.png").exists()
    plt.close(fig)


def test_verbose_save_and_load_prints_to_terminal(tmp_path, capsys):
    io = IO(path=str(tmp_path), verbose=True)
    data = {"a": 1}
    io.save_json(data, "data.json")
    saved_output = capsys.readouterr().out
    assert f"Saved to {tmp_path / 'data.json'}" in saved_output

    io.load_json("data.json")
    loaded_output = capsys.readouterr().out
    assert f"Loaded from {tmp_path / 'data.json'}" in loaded_output


def test_new_directory_with_empty_folder_warns_and_dates_the_path(tmp_path):
    # An empty folder name should fall back to a unique, dated folder name
    # and warn the caller that it did so. new_directory only builds a path
    # object (join, no filesystem write), so there is nothing to clean up.
    with pytest.warns(UserWarning):
        io = IO.new_directory(path=tmp_path, folder="")

    today = datetime.date.today().isoformat()
    assert io.path.name.startswith(today)
