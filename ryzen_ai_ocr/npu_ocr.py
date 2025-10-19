# 导入 ONNX 包
from PIL import Image
import numpy as np
import time
import pathlib

def benchmark_model(session, runs=1000):
    input_shape = session.get_inputs()[0].shape
    input_shape = tuple(1 if isinstance(dim, str) else dim for dim in input_shape)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    start_time = time.time()
    for _ in range(runs):
        outputs = session.run(None, {session.get_inputs()[0].name: input_data})
    end_time = time.time()
    avg_time = (end_time - start_time) / runs
    print('Average inference time over {} runs: {} ms'.format(runs, avg_time * 1000))

def resize_norm_img(img, image_shape=(3, 48, 320)):
    imgC, imgH, imgW = image_shape
    h = imgH
    w = int(img.width * (h / img.height))
    w = min(w, imgW)
    img = img.resize((w, h), Image.BILINEAR)
    padded_img = Image.new('RGB', (imgW, imgH), (0, 0, 0))
    padded_img.paste(img, (0, 0))
    img = np.array(padded_img).astype('float32') / 255.
    img = (img - 0.5) / 0.5
    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    return img

def load_dict(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def ctc_decode(preds, char_list):
    pred_indices = np.argmax(preds, axis=2)[0]
    prev_idx = -1
    text = ''
    for idx in pred_indices:
        if idx != prev_idx and idx < len(char_list) and idx != 0:
            text += char_list[idx]
        prev_idx = idx
    return text

import os
import shutil
import onnxruntime
import onnxruntime_extensions

install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'strix', 'AMD_AIE2P_4x4_Overlay.xclbin')

# print("ONNX Runtime version:", onnxruntime.__version__)
# print("Available providers:", onnxruntime.get_available_providers())
#  日志输出到文件
# import sys
# log_file = open("runtime_output.log", "w", encoding="utf-8")
# sys.stdout = log_file

cache_dir = os.path.abspath('my_cache_dir')
cache_key   = pathlib.Path(r'./inference_quant.onnx').stem
config_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'vaip_config.json')
# config_file = 'vaip_config.json'
options = onnxruntime.SessionOptions()
options.log_severity_level = 0   # 0 = VERBOSE
session = onnxruntime.InferenceSession(
    "inference_quant.onnx",
    providers=[
        'VitisAIExecutionProvider',
        # "DmlExecutionProvider",
        # "CPUExecutionProvider",
    ],
    # providers=['CPUExecutionProvider'],
    sess_options=options,
    provider_options = [{
                "config_file": config_file,
                "cache_dir": cache_dir,
                "cache_key": cache_key,
                "enable_cache_file_io_in_mem":0,
                # 'xclbin': xclbin_file
            }]
)
# print("Session Providers:", session.get_providers())
for i in range (0,1): 
    # benchmark_model(aie_session)
    # 数据预处理
    img = Image.open("text.png").convert("RGB")
    data = resize_norm_img(img)

    # 使用 ONNXRuntime 推理
    input_name = session.get_inputs()[0].name
    result, = session.run(None, {input_name: data})
    print(result)
    # 加载字典
    char_list = [''] + load_dict('ppocrv5_dict.txt')

    # 推理结果后处理
    if result.shape[1] > result.shape[2]:
        result = np.transpose(result, (0, 2, 1))

    text = ctc_decode(result, char_list)
    print("识别结果：", text)