import onnxmltools
from keras.models import load_model

# Update the input name and path for your Keras model
input_keras_model = 'tf-pallet-detection.h5'

# Change this path to the output name and path for the ONNX model
output_onnx_model = 'tf-pallet-detection.onnx'

# Load your Keras model
keras_model = load_model(input_keras_model)

# Convert the Keras model into ONNX
onnx_model = onnxmltools.convert_keras(keras_model)

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, output_onnx_model)