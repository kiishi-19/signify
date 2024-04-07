# pip install 'keras<3.0.0' mediapipe-model-maker
from google.colab import files
import os
import tensorflow as tf
assert tf.__version__.startswith('2')
from mediapipe_model_maker import gesture_recognizer
from mediapipe_model_maker.python.core.utils import quantization as quant
import matplotlib.pyplot as plt

dataset_path = '../data/hello'

data_loader = gesture_recognizer.DataLoader(dataset_path)

model = gesture_recognizer.GestureRecognizer.create(
    train_data=data_loader,
    options=gesture_recognizer.ModelOptions()
)

evaluation_results = model.evaluate(data_loader)
print(evaluation_results)

model.export_model('/saved_models/model')

model.export_model(
    '/path/to/save/quantized_model',
    quant.QuantizationConfig()
)

inference_results = model.inference(data_loader)
print(inference_results)