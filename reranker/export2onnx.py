from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_name = "hotchpotch/japanese-reranker-xsmall-v2"
save_dir = "onnx_model/"

# 自动导出 ONNX（如果已经是 ONNX，会重新导出）
model = ORTModelForSequenceClassification.from_pretrained(
    model_name,
    export=True,      # 这里触发 ONNX 导出
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 保存到目录
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
