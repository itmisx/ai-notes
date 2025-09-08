import os
import cv2
import numpy as np
import onnxruntime as ort
from shapely.geometry import Polygon
import pyclipper

# ==================== DBPostProcess ====================
class DBPostProcess:
    def __init__(self, thresh=0.3, box_thresh=0.5, max_candidates=1000, unclip_ratio=2.0):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap*255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        scores = []
        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            if points.shape[0] < 3:  # 至少三点
                continue
            score = self.box_score_fast(pred, points)
            if score < self.box_thresh:
                continue
            box = self.unclip(points)
            if box is None or len(box) < 3:
                continue
            box = np.array(box)
            box[:,0] = np.clip(np.round(box[:,0] / width * dest_width), 0, dest_width)
            box[:,1] = np.clip(np.round(box[:,1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def box_score_fast(self, bitmap, box):
        h, w = bitmap.shape
        box = np.array(box)
        xmin = np.clip(np.floor(box[:,0].min()).astype(int), 0, w-1)
        xmax = np.clip(np.ceil(box[:,0].max()).astype(int), 0, w-1)
        ymin = np.clip(np.floor(box[:,1].min()).astype(int), 0, h-1)
        ymax = np.clip(np.ceil(box[:,1].max()).astype(int), 0, h-1)
        mask = np.zeros((ymax-ymin+1, xmax-xmin+1), dtype=np.uint8)
        box[:,0] -= xmin
        box[:,1] -= ymin
        cv2.fillPoly(mask, box.reshape(1,-1,2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]

    def unclip(self, box):
        poly = Polygon(box)
        if poly.area == 0 or poly.length == 0:
            return None
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if len(expanded) == 0:
            return None
        return np.array(expanded[0])

    def __call__(self, pred, src_shape):
        if isinstance(pred, np.ndarray) and pred.ndim == 4:
            pred = pred[0,0,:,:]
        segmentation = pred > self.thresh
        src_h, src_w = src_shape
        boxes, scores = self.boxes_from_bitmap(pred, segmentation, src_w, src_h)
        return boxes, scores

# ==================== 裁剪保存函数 ====================
def crop_and_save(img, boxes, save_dir="crops"):
    os.makedirs(save_dir, exist_ok=True)
    for i, box in enumerate(boxes):
        pts = np.array(box, dtype=np.float32)

        # 使用 minAreaRect 获取旋转矩形
        rect = cv2.minAreaRect(pts)
        center, size, angle = rect[0], rect[1], rect[2]
        w, h = int(size[0]), int(size[1])
        if w == 0 or h == 0:
            continue

        # 旋转整张图
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # 裁剪旋转后的矩形
        x1 = int(center[0] - w/2)
        y1 = int(center[1] - h/2)
        x2 = int(center[0] + w/2)
        y2 = int(center[1] + h/2)
        crop = rotated[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        save_path = os.path.join(save_dir, f"crop_{i}.png")
        cv2.imwrite(save_path, crop)
        print(f"保存: {save_path}")

# ==================== 预处理函数 ====================
def resize_norm_img(img, max_side_len=960):
    h, w, _ = img.shape
    ratio = 1.0
    if max(h, w) > max_side_len:
        ratio = max_side_len / max(h, w)
    resize_h = int(h*ratio)
    resize_w = int(w*ratio)
    resize_h = max(32, resize_h//32*32)
    resize_w = max(32, resize_w//32*32)
    resized = cv2.resize(img, (resize_w, resize_h)).astype("float32") / 255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std = np.array([0.229,0.224,0.225], dtype=np.float32)
    resized = (resized - mean)/std
    ratio_h = resize_h / h
    ratio_w = resize_w / w
    return resized, (ratio_h, ratio_w)
def resize_rec_img(img, image_shape=(3,48,320)):
    imgC,imgH, imgW = image_shape
    h = imgH
    w = int(img.shape[1] * (h / img.shape[0]))
    w = min(w, imgW)
    print(h,w)
    resized = cv2.resize(img, (w,h))
    padded = np.zeros((imgH,imgW,3), dtype=np.uint8)
    padded[:, :w] = resized
    img = padded.astype("float32") / 255.
    img = (img-0.5)/0.5
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)
    return img
# -----------------------
# 字典与解码
# -----------------------
def load_dict(dict_path):
    with open(dict_path,'r',encoding='utf-8') as f:
        return [line.strip() for line in f]

def ctc_decode(preds, char_list, blank_idx='first'):
    if blank_idx == 'last':
        blank = preds.shape[2]-1
    else:
        blank = 0
    pred_indices = np.argmax(preds, axis=2)[0]
    prev_idx = -1
    text = ''
    for idx in pred_indices:
        if idx != prev_idx and idx != blank and idx < len(char_list):
            text += char_list[idx]
        prev_idx = idx
    return text
# ==================== 主程序 ====================
def main():
    image_path = "jp.jpg"
    onnx_path = "models/det/inference.onnx"

    img = cv2.imread(image_path)
    src_h, src_w = img.shape[:2]

    resized_img, (ratio_h, ratio_w) = resize_norm_img(img)
    input_tensor = resized_img.transpose(2,0,1)[np.newaxis, :]  # NCHW

    # ONNX 推理
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    pred = outputs[0]  # [1,1,H,W]

    rec_sess = ort.InferenceSession("models/rec/inference.onnx")
    rec_input_name = rec_sess.get_inputs()[0].name


    # 保存概率图
    cv2.imwrite("prob_map.png", (pred[0,0]*255).astype(np.uint8))

    # 后处理
    post = DBPostProcess(thresh=0.2, box_thresh=0.2)  # 阈值可调
    boxes, scores = post(pred, (src_h, src_w))
    boxes = sorted(boxes, key=lambda b: min(p[1] for p in b))
    print(f"检测到 {len(boxes)} 个文本框")

    # 可视化框
    vis = img.copy()
    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0,255,0), 2)
    cv2.imwrite("debug_boxes.png", vis)

    texts = []
    # 加载字典
    char_list = [''] + load_dict('ppocrv5_dict.txt')

    # 识别每行文字
    for i, box in enumerate(boxes):
        pts = np.array(box, dtype=np.float32)

        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        # 裁剪保存
        crop_and_save(img, boxes, save_dir="crops")


        rec_input = resize_rec_img(crop)
        rec_out, = rec_sess.run(None, {rec_input_name: rec_input})

        if rec_out.shape[1] > rec_out.shape[2]:
            rec_out = np.transpose(rec_out, (0,2,1))

        text = ctc_decode(rec_out, char_list, blank_idx='first')
        print(f"识别结果: '{text}'")
        texts.append(text)

    print("\n最终识别结果：", texts)    


if __name__ == "__main__":
    main()
