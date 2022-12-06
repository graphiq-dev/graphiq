# an object of WSGI application
from flask import Flask, jsonify, render_template, json, request, redirect, url_for
import os
from src.utils.draw import Painter
from src.circuit import CircuitDAG

app = Flask(__name__)  # Flask constructor


# A decorator used to tell the application
# which URL is associated function
cache = {}


@app.route('/')
def index():
    data_path = os.path.join(app.root_path, "static", "data")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    visualization_info_path = os.path.join(data_path, "visualization_info.json")
    if not os.path.exists(visualization_info_path):
        f = open(visualization_info_path, "x")
        f.close()
        painter = Painter()
        data = painter.build_visualization_info()
        json_url = os.path.join(app.root_path, "static", "data", "visualization_info.json")
        with open(json_url, "w") as outfile:
            outfile.write(json.dumps(data, indent=3))

    json_url = os.path.join(app.root_path, "static", "data", "visualization_info.json")
    data = json.load(open(json_url))

    return render_template("index.html", visualization_info=data)


@app.route('/circuit_data', methods=['GET', 'POST'])
def circuit_data():
    if request.method == 'GET':
        json_url = os.path.join(app.root_path, "static", "data", "visualization_info.json")
        data = json.load(open(json_url))

        return data
    else:
        data = json.loads(request.get_data())
        json_url = os.path.join(app.root_path, "static", "data", "visualization_info.json")
        with open(json_url, "w") as outfile:
            outfile.write(json.dumps(data, indent=3))

        return render_template("index.html", visualization_info=data)


if __name__ == '__main__':
    app.run(debug=True)


