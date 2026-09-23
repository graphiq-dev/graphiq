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
r"""Guard against invalid escape sequences in the source tree.

Every ``*.py`` file under the directories in ``SWEPT_DIRECTORIES`` is compiled
with all warnings enabled, and any compiler warning whose message contains
"invalid escape sequence" is collected. Python 3.12 reports an invalid escape
(e.g. ``"\s"``) as a ``SyntaxWarning``, older versions as a
``DeprecationWarning``; matching on the message covers both.
"""

import warnings
from pathlib import Path

SWEPT_DIRECTORIES = ("graphiq", "tests", "examples", "visualization")


def test_no_invalid_escape_sequences():
    root = Path(__file__).resolve().parents[1]
    for directory in SWEPT_DIRECTORIES:
        assert (root / directory).is_dir(), f"swept directory missing: {directory}"

    files = sorted(
        path
        for directory in SWEPT_DIRECTORIES
        for path in (root / directory).rglob("*.py")
    )
    assert files, "no .py files found under the swept directories"

    findings = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(source, str(path), "exec")
        for w in caught:
            if "invalid escape sequence" in str(w.message):
                findings.append((str(path.relative_to(root)), w.lineno, str(w.message)))

    assert not findings, "\n".join(str(entry) for entry in findings)
