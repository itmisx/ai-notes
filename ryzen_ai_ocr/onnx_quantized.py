import os
import cv2
import onnx
import copy
import numpy as np
import onnxruntime
from onnx import helper, numpy_helper
from onnxruntime.quantization import CalibrationDataReader

def get_model_input_name(input_model_path: str) -> str:
    model = onnx.load(input_model_path)
    model_input_name = model.graph.input[0].name
    return model_input_name

class ImageDataReader(CalibrationDataReader):

    def __init__(self, calibration_image_folder: str, input_name: str):
        self.enum_data = None

        self.input_name = input_name

        self.data_list = self._preprocess_images(
                calibration_image_folder)

    def _preprocess_images(self, image_folder: str):
        data_list = []
        img_names = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
        for name in img_names:
            input_image = cv2.imread(os.path.join(image_folder, name))
            # Resize the input image. Because the size of Resnet50 is 224.
            input_image = cv2.resize(input_image, (48, 320))
            input_data = np.array(input_image).astype(np.float32)
            input_data = input_data.transpose(2, 0, 1)
            # Custom Pre-Process
            input_size = input_data.shape
            if input_size[1] > input_size[2]:
                input_data = input_data.transpose(0, 2, 1)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = input_data / 255.0
            data_list.append(input_data)

        return data_list

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: data} for data in self.data_list])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

model = onnx.load("models/inference_fixed.onnx")
#  修改maxPool算子
for node in model.graph.node:
    if node.op_type in ["MaxPool"]:
        attrs = {a.name: a for a in node.attribute}
        if "ceil_mode" in attrs:
            # auto_pad_value = attrs["auto_pad"].s.decode("utf-8")
            node.attribute.remove(attrs["ceil_mode"])
            node.attribute.append(helper.make_attribute("ceil_mode", 0))
        if "auto_pad" in attrs:
            # auto_pad_value = attrs["auto_pad"].s.decode("utf-8")
            # 删除 auto_pad
            node.attribute.remove(attrs["auto_pad"])
            # 简单示例：用 0-padding 替代
            pads = [1, 1, 0, 0]  # [top, left, bottom, right]
            # 也可根据卷积核、stride动态推算（见下方）
            node.attribute.append(helper.make_attribute("pads", pads))
onnx.save(model, "models/inference_shape.onnx")

from quark.onnx.quantization.config import QConfig
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config.spec import BFloat16Spec,QLayerConfig


# Set up quantization with a specified configuration
# For example, use "A8W8" for Ryzen AI INT8 quantization

# quant_config = get_default_config("XINT8")
# quant_config.extra_options["BF16QDQToCast"] = True
# config = Config(global_quant_config=quant_config)

input_model_path = "models/inference_shape.onnx"
quantized_model_path = "models/inference_quant.onnx"
calib_data_path = "calib_data_1"
model_input_name = get_model_input_name(input_model_path)
calib_data_reader = ImageDataReader(calib_data_path, model_input_name)

# quant_config = QConfig.get_default_config("BF16")
# quant_config.global_quant_config.extra_options["BF16QDQToCast"] = True
# quant_config.global_quant_config.extra_options["WeightScaled"] = True
# quant_config.global_quant_config.extra_options["ActivationScaled"] = True

# activation_spec = BFloat16Spec()
# weight_spec = BFloat16Spec()
# quant_config = QConfig(
#     global_config=QLayerConfig(activation=activation_spec, weight=weight_spec),
#     WeightScaled=True, 
#     ActivationScaled=True,
#     BF16QDQToCast=True,
# )

quant_config = QConfig.get_default_config("XINT8")
# quant_config.global_quant_config.extra_options["AlignReshape"] = True
quant_config.global_quant_config.op_types_to_quantize=[
    "Conv",
    "Relu",
    "Concat",
    "AveragePool",
    "Sigmoid",
    "Mul",
    "Transpose",
    "LayerNormalization",
    "Softmax",
    "Squeeze",
    "UnSqueeze",
    "MaxPool",
    "MatMul",
    "Add",
    "Slice",
    # mobile
    # "Conv",
    # "Mul",
    # "Add",
    # "HardSwish",
    # "GlobalAveragePool",
    # "Relu",
    # "HardSigmoid",
    # "AveragePool",
    # "Unsqueeze",
    # "Squeeze",
    # "Sigmoid",
    # "Reshape",
    # "Transpose",
    # "LayerNormalization",
    # "MatMul",
    # # "Slice",
    # "Softmax",
    # "Concat", 
]
# insert_reshape_around_slice("models/inference_shape.onnx", "models/inference_shape.onnx")
# quant_config.global_quant_config.enable_npu_cnn = False
# quant_config.global_quant_config.enable_npu_transformer = False
quantizer = ModelQuantizer(quant_config)
quantizer.quantize_model(input_model_path, quantized_model_path,calib_data_reader)
