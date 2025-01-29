from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, List

import os

app = FastAPI()

# Load your model
try:
    literacy_model  = joblib.load("app/domains/EIP/domain1-literacy-model.pkl")
    numeracy_model  = joblib.load("app/domains/EIP/domain2-numeracy-model.pkl")
    fine_motor_model   = joblib.load("app/domains/EIP/domain3-finemotor-model.pkl")
    social_skills_model   = joblib.load("app/domains/EIP/domain4-socialskills-model.pkl")
except FileNotFoundError:
    raise RuntimeError("Model file not found.")

class PredictProgress(BaseModel):
    domains: Dict[str, List[int]] 

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Model API!"}

@app.post("/eip/student_progress")
def predict(request: PredictProgress):
    try:
        print("Received data:", request.domains)

        predictions = {}

        # Process each domain individually
        for domain, evaluations in request.domains.items():
            if len(evaluations) != 3:
                return {"error": f"Invalid data length for {domain}, expected 3 values"}

            input_data = np.array(evaluations).reshape(1, 3)  # Convert list to np.array
            if domain == "Literacy":
                prediction = literacy_model.predict(input_data)
            elif domain == "Numeracy":
                prediction = numeracy_model.predict(input_data)
            elif domain == "Fine Motor":
                prediction = fine_motor_model.predict(input_data)
            elif domain == "Social Skills":
                prediction = social_skills_model.predict(input_data)
            else:
                return {"error": f"Unknown domain {domain}"}

            predictions[domain] = int(prediction[0])

        print("Predictions:", predictions)
        return {"predictions": predictions}

    except Exception as e:
        print("Error during prediction:", str(e))
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    PORT = os.getenv("PORT", 8000)
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))