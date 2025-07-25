> 该用例尝试，进行将padddleocr的模型转为onnx的格式，并运行


#### 模型转换
paddleocr4版本之后，下载的模型中已经没有padmodel文件，取而代之的是inference.json,不能通过paddle2onnx命令直接转换,需要使用paddlex指令
```bash
# Windows 用户需使用以下命令安装 paddlepaddle dev版本
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
# 再安装paddle2onnx插件
paddlex --install paddle2onnx
# 执行转换
# paddle_model_dir paddle模型路径
# onnx_model_dir 转换onnx的模型路径
paddlex  --paddle2onnx --paddle_model_dir /your/paddle_model/dir  --onnx_model_dir /your/onnx_model/output/dir --opset_version 7
# 模型下载地址
# https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/module_usage/text_recognition.html#_2
# 如果进行文字识别，需要单独下载字库，需要模型训练的字库一致
# 字库下载地址 https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/ppocrv5_dict.txt