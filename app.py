# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load initial dataset and model
df_1 = pd.read_csv("tel_churn.csv")
model = pickle.load(open("model.sav", "rb"))

# Define route for homepage
@app.route("/")
def loadPage():
    return render_template('home.html', query="")

# Define route for prediction
@app.route("/", methods=['POST'])
def predict():
    # Get form inputs
    inputQuery = [request.form[f'query{i}'] for i in range(1, 20)]

    # Create a DataFrame from input
    columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
               'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
               'PaymentMethod', 'tenure']
    data = pd.DataFrame([inputQuery], columns=columns)

    # Convert appropriate columns to numeric
    for col in ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'tenure']:
        data[col] = pd.to_numeric(data[col])

    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    data['tenure_group'] = pd.cut(data.tenure, bins=range(1, 80, 12), right=False, labels=labels)
    
    # Drop the original 'tenure' column
    data.drop(columns=['tenure'], inplace=True)
    
    # Get dummy variables
    data_dummies = pd.get_dummies(data)
    
    # Ensure the new data frame has all the columns the model expects
    model_columns = pd.get_dummies(df_1).drop(columns='Churn').columns
    for col in model_columns:
        if col not in data_dummies:
            data_dummies[col] = 0
    
    # Reorder columns to match training data
    data_dummies = data_dummies[model_columns]

    # Make predictions
    prediction = model.predict(data_dummies)
    probability = model.predict_proba(data_dummies)[:, 1]
    
    if prediction == 1:
        output1 = "This customer is likely to be churned!"
        output2 = f"Confidence: {probability[0] * 100:.2f}%"
    else:
        output1 = "This customer is likely to continue!"
        output2 = f"Confidence: {probability[0] * 100:.2f}%"

    return render_template('home.html', output1=output1, output2=output2, **request.form)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
