# an object of WSGI application
from flask import Flask, jsonify, render_template, json, request, redirect, url_for
import os
from visualization_main import Painter

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

    painter = Painter()

    painter.add_register(register=0, reg_type="p")
    painter.add_register(register=1, reg_type="p")
    painter.add_register(register=2, reg_type="p")
    painter.add_register(register=3, reg_type="p")
    painter.add_register(register=0, reg_type="e")
    painter.add_register(register=0, reg_type="c")
    painter.add_gate(gate_name="H", qargs=["e0"])
    painter.add_gate(gate_name="CX", qargs=["e0", "p1"])
    painter.add_gate(gate_name="CX", qargs=["e0", "p2"])

    visualization_info = painter.build_visualization_info()

    return render_template("index.html",
        data=cache["circuit"],
        draw=cache["circuit"]["openqasm"],
        visualization_info=visualization_info,
    )


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


