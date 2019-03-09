import tensorflow as tf
      
interpreter = tf.contrib.lite.Interpreter(model_path="your_path")
interpreter.allocate_tensors()

print("input")
print(interpreter.get_input_details()[0])

print("output")
print(interpreter.get_output_details()[0])
