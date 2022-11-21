# an object of WSGI application
from flask import Flask, jsonify, render_template
from flask import session

app = Flask(__name__)  # Flask constructor

# A decorator used to tell the application
# which URL is associated function
@app.route('/')
def index():
    test_data = {
        "t": 1,
        "z": 2,
    }

    return render_template("index.html", test_data=test_data)


if __name__ == '__main__':
    app.run(debug=True)


