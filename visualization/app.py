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
# an object of WSGI application
from flask import Flask, jsonify, render_template, json, request, redirect, url_for

app = Flask(__name__)  # Flask constructor

# A decorator used to tell the application
# which URL is associated function
cache = {
    "circuit_data": {},
}


@app.route("/")
def index():
    return render_template("index.html", visualization_info=cache["circuit_data"])


@app.route("/circuit_data", methods=["GET", "POST"])
def circuit_data():
    if request.method == "GET":
        return cache["circuit_data"]
    else:
        data = json.loads(request.get_data())
        cache["circuit_data"] = data

        return render_template("index.html", visualization_info=cache["circuit_data"])


if __name__ == "__main__":
    app.run()
