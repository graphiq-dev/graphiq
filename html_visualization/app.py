# an object of WSGI application
from flask import Flask, jsonify, render_template, json, request, redirect, url_for
import os

from draw import DrawingManager

app = Flask(__name__)  # Flask constructor


# A decorator used to tell the application
# which URL is associated function
cache = {}


@app.route("/create")
def create():
    cache["foo"] = 0
    return cache


@app.route("/increment")
def increment():
    cache["foo"] = cache["foo"] + 1
    return cache


@app.route("/read")
def read():
    return cache


@app.route('/')
def index():
    json_url = os.path.join(app.root_path, "static", "data", "circuit_data.json")
    cache["circuit"] = json.load(open(json_url))
    draw = DrawingManager(cache["circuit"]["openqasm"])

    return render_template("index.html", data=cache["circuit"], draw=draw)


@app.route('/circuit_data', methods=['GET', 'POST'])
def circuit_data():
    if request.method == 'GET':
        json_url = os.path.join(app.root_path, "static", "data", "circuit_data.json")
        data = json.load(open(json_url))

        return data
    else:
        data = json.loads(request.get_data())
        json_url = os.path.join(app.root_path, "static", "data", "circuit_data.json")
        with open(json_url, "w") as outfile:
            outfile.write(json.dumps(data, indent=3))

        return data


@app.route('/get_circuit_position_data')
def get_circuit_position_data():
    position_data = {}

    return jsonify(position_data)


if __name__ == '__main__':
    app.run(debug=True)


