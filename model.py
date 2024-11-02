import joblib

class StrokePredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        self.input_data = input_data
        prediction = self.model.predict(input_data)  # input_data should be 2D
        risk_percentage = self.calculate_risk_percentage(prediction)
        return prediction[0], risk_percentage
    
    def calculate_risk_percentage(self, prediction):
        # Example: Calculate risk percentage based on prediction output
        risk = self.model.predict_proba(self.input_data)
        positive_probability = risk[0][1]
        return positive_probability
