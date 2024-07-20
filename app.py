# Importing essential libraries
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the model
model = pickle.load(open('loan_prediction.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Collecting form data
        Age = float(request.form.get('Age', 0))
        Income = float(request.form.get('Income', 0))
        LoanAmount = float(request.form.get('LoanAmount', 0))
        CreditScore = float(request.form.get('CreditScore', 0))
        NumCreditLines = int(request.form.get('NumCreditLines', 0))
        DTIRatio = float(request.form.get('DTIRatio', 0))
        Education = int(request.form.get('Education', 0))
        EmploymentType = int(request.form.get('EmploymentType', 0))
        MaritalStatus = int(request.form.get('MaritalStatus', 0))
        HasMortgage = int(request.form.get('HasMortgage', 0))
        HasDependents = int(request.form.get('HasDependents', 0))
        LoanPurpose = int(request.form.get('LoanPurpose', 0))
        HasCoSigner = int(request.form.get('HasCoSigner', 0))

        # Preparing data for prediction
        data = np.array([[Age, Income, LoanAmount, CreditScore, NumCreditLines, DTIRatio, Education,
                          EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner]])

        # Making prediction
        my_prediction = model.predict(data)
        
        # Return prediction result
        return render_template('result.html', prediction=my_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

