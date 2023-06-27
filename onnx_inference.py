import cv2
import onnxruntime as ort
import numpy as np
import time


onnx_model_path = "./out.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
onnx_input_name = ort_session.get_inputs()[0].name
onnx_outputs_names = ort_session.get_outputs()[0].name
img_path = "test_img/0004x3.png"
img = cv2.imread(img_path)
img = np.asarray(img, np.float32)/255.0
img = img.transpose((2, 0, 1))
img = img[np.newaxis, :, :, :]
for i in range(100):
    start = time.time()

    onnx_result = ort_session.run([onnx_outputs_names], input_feed={onnx_input_name: img})[0]
    print(time.time()-start)
onnx_result = onnx_result.squeeze()*255
onnx_result = onnx_result.transpose((1, 2, 0))
cv2.imwrite("out_images/"+img_path.split('/')[-1], onnx_result)