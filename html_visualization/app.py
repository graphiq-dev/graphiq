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
