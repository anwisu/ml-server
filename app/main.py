from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, List

import os

app = FastAPI()

# Load your model
try:
    domain1 = joblib.load("app/domains/domain1.pkl");
    domain2 = joblib.load("app/domains/domain2.pkl");
    domain3 = joblib.load("app/domains/domain3.pkl");
    domain4 = joblib.load("app/domains/domain4.pkl");
except FileNotFoundError:
    raise RuntimeError("Model file not found.")

class PredictProgress(BaseModel):
    domains: Dict[str, List[int]] 

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Model API!"}

@app.post("/predict/student_progress")
def predict(request: PredictProgress):
    try:
        predictions = {}
        
        # Process each domain individually
        for domain, evaluations in request.domains.items():
            input_data = np.array(evaluations).reshape(1, 2)  # Convert list to np.array
            if domain == "Domain_1":
                prediction = domain1.predict(input_data)
            elif domain == "Domain_2":
                prediction = domain2.predict(input_data)
            elif domain == "Domain_3":
                prediction = domain3.predict(input_data)
            elif domain == "Domain_4":
                prediction = domain4.predict(input_data)
            predictions[domain] = int(prediction[0])
        
        return {"predictions": predictions}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    PORT = os.getenv("PORT", 8000)
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))