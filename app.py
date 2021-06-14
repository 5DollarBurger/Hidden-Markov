from flask import Flask, request, json
import numpy as np
from src.HMMDiscrete import HMMDiscrete

app = Flask(__name__)

@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        pass

    return "Welcome to Hidden Markovian Training."

@app.route("/update", methods=["GET", "POST"])
def update():
    if request.method == "POST":
        inputDict = request.get_json()
        pi = np.array(inputDict["initMat"])
        A = np.array(inputDict["transMat"])
        B = np.array(inputDict["emitMat"])
        X = np.array(inputDict["seqList"])

        ins = HMMDiscrete(pi=pi, A=A, B=B)
        costList = ins.update(X=X)

        resp = {
            "isValid": True,
            "errorList": [],
            "result": {
                "initMat": ins.pi.tolist(),
                "transMat": ins.A.tolist(),
                "emitMat": ins.B.tolist(),
                "costList": costList
            }
        }
        return resp

    return "Welcome to Hidden Markovian Updating."

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        inputDict = request.get_json()
        pi = np.array(inputDict["initMat"])
        A = np.array(inputDict["transMat"])
        B = np.array(inputDict["emitMat"])
        x = np.array(inputDict["seq"])

        ins = HMMDiscrete(pi=pi, A=A, B=B)
        z = ins.predict(x=x)

        resp = {
            "isValid": True,
            "errorList": [],
            "result": z.tolist()
        }

        return resp

    return "Welcome to Hidden Markovian Predictions."


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
    # app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)