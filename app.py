from flask import Flask, render_template, request
from sklearn.externals import joblib

#import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

#mul_reg = open("Revenue_lm_pkl_16042019.pkl", "rb")
#ml_model = joblib.load(mul_reg)

@app.route("/")
def home():
    return render_template('home.html')
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("I was here 1")
    if request.method == 'POST':
        #print(request.form.get('IP_TOTAL'))
        try:
            IP_TOTAL = float(request.form['IP_TOTAL'])
            OP_TOTAL = float(request.form['OP_TOTAL'])
            TOTAL_PHARMACY_REVENUE = float(request.form['TOTAL_PHARMACY_REVENUE'])
            OP_Lab_Revenue = float(request.form['OP_Lab_Revenue'])
            IP_Lab_Revenue = float(request.form['IP_Lab_Revenue'])
#            Market_Spend = float(request.form['Market_Spend'])
            pred_args = [IP_TOTAL, OP_TOTAL, TOTAL_PHARMACY_REVENUE, OP_Lab_Revenue, IP_Lab_Revenue]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            mul_reg = open("Revenue_lm_pkl_16042019.pkl", "rb")
            ml_model = joblib.load(mul_reg)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction)


if __name__ == "__main__":
    app.run(debug=True)
