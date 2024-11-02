from flask import Flask, request, jsonify
from model import StrokePredictor
import pandas as pd

app = Flask(__name__)
predictor = StrokePredictor('stroke_predictor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the incoming request
    user_data = request.json
    
    # Create a dictionary that matches the expected format for prediction
    sample_data = {
        "gender": user_data.get("gender"),  # 1 for Female, 0 for Male
        "age": user_data.get("age"),  
        "hypertension": user_data.get("hypertension"),  
        "heart_disease": user_data.get("heart_disease"),  
        "ever_married": user_data.get("ever_married"),
        "avg_glucose_level": user_data.get("avg_glucose_level"), 
        "bmi": user_data.get("bmi"),  
        "work_type_Govt_job": 1 if user_data.get("work_type") == "Govt_job" else 0,  
        "work_type_Private": 1 if user_data.get("work_type") == "Private" else 0,  
        "work_type_Self-employed": 1 if user_data.get("work_type") == "Self-employed" else 0,  
        "work_type_children": 1 if user_data.get("work_type") == "children" else 0,  
        "Residence_type_Rural": 1 if user_data.get("Residence_type") == "Rural" else 0,  
        "Residence_type_Urban": 1 if user_data.get("Residence_type") == "Urban" else 0,  
        "smoking_status_Unknown": 1 if user_data.get("smoking_status") == "Unknown" else 0,  
        "smoking_status_formerly smoked": 1 if user_data.get("smoking_status") == "formerly smoked" else 0,  
        "smoking_status_never smoked": 1 if user_data.get("smoking_status") == "never smoked" else 0, 
        "smoking_status_smokes": 1 if user_data.get("smoking_status") == "smokes" else 0  
    }

    # Convert the processed sample data into a DataFrame
    sample_df = pd.DataFrame([sample_data])

    # Debugging: Print input data and shape
    print("Input Data for Prediction:")
    print(sample_df)

    # Predict with the model
    try:
        prediction = predictor.predict(sample_df)  # Ensure we pass a DataFrame with the correct shape
        risk_percentage = predictor.calculate_risk_percentage(prediction) * 100  # Assuming binary output for stroke prediction

        return jsonify({
            'prediction': int(prediction[0]),  # Convert prediction to int for JSON response
            'risk_percentage': f"{risk_percentage:.2f}",  # Format as percentage
            'message': determine_health_message(risk_percentage)  # Call function to get a health message
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def determine_health_message(risk_percentage):
    if risk_percentage <= 10:
        return "Excellent! Your health looks great. Keep maintaining your current lifestyle and habits!"
    elif 10 < risk_percentage <= 20:
        return "Very Good! You have a low risk of stroke. Continue your healthy habits and stay active!"
    elif 20 < risk_percentage <= 30:
        return "Good! You're doing well, but remember to stay mindful of your health. Regular check-ups are advised."
    elif 30 < risk_percentage <= 40:
        return "Moderate risk. It's time to pay closer attention to your health. Consider consulting a healthcare professional for advice."
    elif 40 < risk_percentage <= 50:
        return "Caution! Your risk of stroke is higher than average. Evaluate your lifestyle choices and seek medical guidance."
    elif 50 < risk_percentage <= 60:
        return "High risk. Please take immediate action to improve your health. Consult with a healthcare provider for a personalized plan."
    elif 60 < risk_percentage <= 70:
        return "Very High risk! It is critical to seek medical attention and evaluate your health habits."
    else:
        return "Severe risk! Urgent medical evaluation is necessary. Please consult a healthcare professional immediately."


    # Additional messages can be added as needed


if __name__ == '__main__':
    app.run(debug=True)
