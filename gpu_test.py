import tensorflow as tf
v = tf.__version__
# print(v)

print(tf.config.list_physical_devices('GPU'))


import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current GPU device:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")