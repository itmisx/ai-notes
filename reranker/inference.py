import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

save_dir = "onnx_model/"

query = "感動的な映画について"
candidates = [
    "深いテーマを持ちながらも、観る人の心を揺さぶる名作。登場人物の心情描写が秀逸で、ラストは涙なしでは見られない。",
    "重要なメッセージ性は評価できるが、暗い話が続くので気分が落ち込んでしまった。もう少し明るい要素があればよかった。",
    "どうにもリアリティに欠ける展開が気になった。もっと深みのある人間ドラマが見たかった。",
    "アクションシーンが楽しすぎる。見ていて飽きない。ストーリーはシンプルだが、それが逆に良い。",
]

tokenizer = AutoTokenizer.from_pretrained(save_dir)
session = ort.InferenceSession(f"{save_dir}/model.onnx")

scores = []
for candidate in candidates:
    inputs = tokenizer(query, candidate, return_tensors="np", padding=True, truncation=True)
    onnx_inputs = {k: v for k, v in inputs.items()}
    logit = session.run(None, onnx_inputs)[0]
    prob = 1 / (1 + np.exp(-logit))
    scores.append(prob[0][0])

# 按概率排序
# sorted_candidates = [x for _, x in sorted(zip(scores, candidates), reverse=True)]
# print("Reranked candidates:", sorted_candidates)
print("Scores:",scores)