from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def car_price_prediction():
    if request.method == "POST":
        p1 = float(request.form["Present_Price"])
        p2 = float(request.form["Kms_Driven"])
        p3 = float(request.form["Fuel_Type"])
        p4 = float(request.form["Seller_Type"])
        p5 = float(request.form["Transmission"])
        p6 = float(request.form["Owner"])
        p7 = float(request.form["Age"])

        model = joblib.load('car_price_predictor')
        data_new = pd.DataFrame({
            'Present_Price': [p1],
            'Kms_Driven': [p2],
            'Fuel_Type': [p3],
            'Seller_Type': [p4],
            'Transmission': [p5],
            'Owner': [p6],
            'Age': [p7]
        })
        result = model.predict(data_new)
        return render_template("result.html", result=result[0])

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
