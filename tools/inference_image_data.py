from PIL import Image
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="your_path")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
# 画像データパスとstructure_tfl.pyの実行で出力されたinput shapeの画像サイズを入れる
image = np.array(Image.open('samples/pose1.jpg').resize((192, 192)), dtype=np.float32).reshape(input_shape)

# 画像自体のバイナリデータを透過も含めて表示(rgba)
# print(image)
# 画像のdata typeは透過も含めているため、float32で表現している。
# print(image.dtype)
# 次元数
# print(image.ndim)
# 定義されているinput shape
# print(image.shape)

interpreter.set_tensor(input_details[0]['index'], image)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

np.set_printoptions(threshold=np.inf)
print(output_data)
