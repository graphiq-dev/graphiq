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

import pytest

from graphiq.io import IO


def test_new_directory_with_empty_folder_warns_and_dates_the_path(tmp_path):
    # An empty folder name should fall back to a unique, dated folder name
    # and warn the caller that it did so. new_directory only builds a path
    # object (join, no filesystem write), so there is nothing to clean up.
    with pytest.warns(UserWarning):
        io = IO.new_directory(path=tmp_path, folder="")

    today = datetime.date.today().isoformat()
    assert io.path.name.startswith(today)
