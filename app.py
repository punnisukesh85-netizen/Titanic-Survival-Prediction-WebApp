from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.joblib")
selected_features = joblib.load("selected_features.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    df = pd.DataFrame([{
        'Pclass': int(data['Pclass']),
        'Age': float(data['Age']),
        'Fare': float(data['Fare']),
        'SibSp': int(data['SibSp']),
        'Parch': int(data['Parch']),
        'Sex_male': 1 if data['Sex'] == 'male' else 0,
        'Sex_female': 1 if data['Sex'] == 'female' else 0,
        'Embarked_C': 1 if data['Embarked'] == 'C' else 0,
        'Embarked_Q': 1 if data['Embarked'] == 'Q' else 0,
        'Embarked_S': 1 if data['Embarked'] == 'S' else 0
    }])

    df = df[selected_features]

    prediction = model.predict(df)[0]
    result = "Survived" if prediction == 1 else "Did not survive"

    return render_template("index.html", prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)


