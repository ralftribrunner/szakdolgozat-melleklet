# h5 formátumú modellből képes onnx modelt generálni

import onnxmltools
from keras.models import load_model

input_keras_model = 'tf-pallet-detection.h5'

output_onnx_model = 'tf-pallet-detection.onnx'

keras_model = load_model(input_keras_model)

onnx_model = onnxmltools.convert_keras(keras_model)

onnxmltools.utils.save_model(onnx_model, output_onnx_model)

# Forrás:
# https://github.com/onnx/onnxmltools