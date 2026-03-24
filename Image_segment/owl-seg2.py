import torch
from transformers import pipeline
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import cv2
import numpy as np

# --- 1. 初始化模型 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载 OWL-ViT (这就是你的大脑)
detector = pipeline(
    model="google/owlvit-large-patch14", 
    task="zero-shot-object-detection", 
    device=0 if device.type == "cuda" else -1
)

# 加载 SAM
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth" 
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
sam_predictor = SamPredictor(sam)

# --- 2. 加载与预处理 ---
IMAGE_PATH = "pizza.jpg"
image_pillow = Image.open(IMAGE_PATH).convert("RGB")
image_cv2 = cv2.imread(IMAGE_PATH)
h, w, _ = image_cv2.shape
sam_predictor.set_image(np.array(image_pillow))

# --- 3. OWL-ViT 定位 (设置极低阈值，确保不落空) ---
candidate_labels = ["pizza", "aluminum foil"]
predictions = detector(image_pillow, candidate_labels=candidate_labels)

# --- 4. SAM 分割 + 凸包闭合 ---
result_img = image_cv2.copy()

for pred in predictions:
    if pred["score"] < 0.01: # 门槛放低，保证比萨能出来
        continue
        
    box = pred["box"]
    # 给框稍微加 5 像素的余量，防止边缘被切
    input_box = np.array([
        max(0, box["xmin"] - 5), 
        max(0, box["ymin"] - 5), 
        min(w, box["xmax"] + 5), 
        min(h, box["ymax"] + 5)
    ])
    
    # SAM 抠图
    masks, _, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    
    mask = masks[0].astype(np.uint8) * 255
    
    # 找到该区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 取最大的那个轮廓
        max_cnt = max(contours, key=cv2.contourArea)
        
        # 【核心改进：凸包闭合】
        # 这一行就是那个“焊条”，它会忽略边缘的小断裂，强制连成一个封闭多边形
        hull = cv2.convexHull(max_cnt)
        
        # 画出绿色封闭轮廓
        cv2.drawContours(result_img, [hull], -1, (0, 255, 0), 4)

# --- 5. 保存结果 ---
cv2.imwrite("owl_sam_closed_result.jpg", result_img)
print("这次 OWL 回来了，边缘也焊死了！请查看 owl_sam_closed_result.jpg")