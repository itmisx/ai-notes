from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import warnings
import os

save_dir = "onnx_model/"
# 抑制警告
# warnings.filterwarnings("ignore")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "hotchpotch/japanese-reranker-xsmall-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# 准备输入示例
text_a = "これは日本語のテキストです。"
text_b = "これは別の文です。"
inputs = tokenizer(text_a, text_b, return_tensors="pt", padding=True, truncation=True, max_length=512)


torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    f"{save_dir}model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
    },
    opset_version=17,
    dynamo=False, 
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)
print("ONNX 模型导出成功！")