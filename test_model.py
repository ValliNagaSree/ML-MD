import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from predict import make_prediction
MODEL_PATH="best_model.joblib"
#fixture 1
@pytest.fixture
def model():
  assert os.path.exists(MODEL_PATH),"Modelfile not found!"
  return joblib.load(MODEL_PATH)
  #fixture 2
@pytest.fixture
def sample_data():
  data=load_breast_cancer()
  X=pd.DataFrame(
    data.data,
    columns=data.feature_names)
  return X.iloc[:5]
#test 1
def test_model_loads(model):
  assert model is not None
#test 2
def test_model_prediction(model,sample_data):
  predictions=model.predict(sample_data)
  assert predictions is not None
#test 3
def test_prediction_shape(model,sample_data):
  predictions=model.predict(sample_data)
  assert len(predictions)==len(sample_data)
#test 4
def test_prediction_values(model,sample_data):
  predictions=model.predict(sample_data)
  assert set(predictions).issubset({0,1})
#test 5
def test_prediction_probabilities(model,sample_data):
  probabilities=model.predict_proba(sample_data)
  assert probabilities.shape==(5,2)
  assert np.all(probabilities>=0)
  assert np.all(probabilities<=1)
  assert np.allclose(probabilities.sum(axis=1),1)
#test 6
def test_model_pipeline(model):
  assert hasattr(model,"named-steps")
  assert "scaler" in model.named_steps
  assert "model" in model.named_steps
#test 7
def test_make_prediction(model,sample_data):
  output=make_prediction(model,sample_data)
  assert "prediction" in output
  assert output["prediction"] in {"Malignant","Benign"}
#test 8
def test_make_prediction_probabilities(model,sample_data):
  output=make_prediction(model,sample_data)
  probabilities=output["probabilities"]
  assert "malignant" in probabilities
  assert "benign" in probabilities
  assert 0<=probabilities["malignant"]<=1
  assert 0<=probabilities["benign"]<=1

