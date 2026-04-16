from flask import Flask, jsonify, request
from flask_cors import CORS
from main_engine import run_full_optimization

app = Flask(__name__)
CORS(app)

@app.route("/optimize", methods=["POST"])
def optimize():

    params = request.json

    result = run_full_optimization(params)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)