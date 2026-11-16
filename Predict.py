#predict.py
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
#Step 1:Load the saved model
def load_model(model_path="best_model.joblib"):
  model=joblib.load(model_path)
  return model
#step 2:create a reuseable prediction function
def make_prediction(model,input_data):
  prediction=model.predict(input_data)
  probabilities=model.predict_proba(input_data)
  result="Malignant" if prediction[0]==0 else "Benign"
  return{
    "prediction":result,
    "probabilities":{
       "malignant":probabilities[0][0],
       "benign":probabilities[0][1]
  }
}
#Step 3:Run a sample prediction when executed directly
 if__name__=="__main__":
   model=load_model()
   data=load_breast_cancer()
   feature_names=data.feature_names
   sample_input=dict(zip(feature_names,data.data[0]))
   sample_input["mean radius"]=15.0
   sample_input["mean texture"]=18.0
   sample_input["mean perimeter"]=100.0
   input_df=pd.DataFrame([sample_input],
      columns=feature_names
)
output=make_prediction(model,input_df)
print("Model loaded successfully!")
print("\n Prediction:",output["prediction"])
print("\n Prediction Probabilities:")
print("Malignant:",output["probabilities"]["malignant"])
print("Benign:",output["probabilities"]["benign"])

