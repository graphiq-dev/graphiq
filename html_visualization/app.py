# an object of WSGI application
from flask import Flask, jsonify, render_template, json
import os

app = Flask(__name__)  # Flask constructor


# A decorator used to tell the application
# which URL is associated function
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/get_test_data')
def get_test_data():
    data = {
        "value1": 1,
        "value2": 2,
    }

    return jsonify(data)


@app.route('/get_circuit_data')
def get_circuit_data():
    json_url = os.path.join(app.root_path, "static", "appdata", "circuit_data.json")
    data = json.load(open(json_url))

    return data


if __name__ == '__main__':
    app.run(debug=True)


