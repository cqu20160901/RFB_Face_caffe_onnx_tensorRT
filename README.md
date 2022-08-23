# Face-Detector_RFB_SSD
RFB320_Face_Detector 部署版本，后处理用python语言和C++语言形式进行改写，便于移植不同平台（caffe、onnx、tensorRT）。

caffe：去除维度变换层的prototxt、caffeModel、测试图像、测试结果、测试demo脚本

onnx：onnx模型、测试图像、测试结果、测试demo脚本

tensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

# 测试结果
![image](https://github.com/cqu20160901/RFB_Face_caffe_onnx_tensorRT/blob/main/caffe/test_result.jpg)

# 参考
[1] https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
