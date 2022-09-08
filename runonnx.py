import onnxruntime as ort
import numpy as np


model = ort.InferenceSession(f"./model_imgs.onnx")

# timeout

x = model.run(None, np.randn(1, 256))

print(x)