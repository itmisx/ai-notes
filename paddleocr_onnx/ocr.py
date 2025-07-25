# 导入 ONNX 包
import onnxruntime
from PIL import Image
import numpy as np

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

# ✅ 使用 DirectML 加速，兼容 AMD GPU/NPU
providers = ['DmlExecutionProvider', 'CPUExecutionProvider']

# 创建 ONNX 推理会话
sess = onnxruntime.InferenceSession("PP-OCRv5_server_rec_infer/inference.onnx", providers=providers)

# 数据预处理
img = Image.open("text.png").convert("RGB")
data = resize_norm_img(img)

# 使用 ONNXRuntime 推理
input_name = sess.get_inputs()[0].name
result, = sess.run(None, {input_name: data})

# 加载字典
char_list = [''] + load_dict('ppocrv5_dict.txt')

# 推理结果后处理
if result.shape[1] > result.shape[2]:
    result = np.transpose(result, (0, 2, 1))

text = ctc_decode(result, char_list)
print("识别结果：", text)
