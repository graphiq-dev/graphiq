# an object of WSGI application
from flask import Flask, jsonify, render_template, json, request, redirect, url_for
import os
from visualization_main import Painter

app = Flask(__name__)  # Flask constructor


# A decorator used to tell the application
# which URL is associated function
cache = {}


@app.route('/')
def index():
    painter = Painter()

    painter.add_register(reg_name="p", size=4)
    painter.add_register(reg_name="e", size=1)
    painter.add_register(reg_name="c", size=4, reg_type="creg")

    painter.add_gate(gate_name="H", qargs=["e0"])
    painter.add_gate(gate_name="H", qargs=["e0"])
    painter.add_gate(gate_name="RX", qargs=["p0"], params={"theta": "pi/2", "phi": "pi/4"})

    visualization_info = painter.build_visualization_info()

    return render_template("index.html",
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


