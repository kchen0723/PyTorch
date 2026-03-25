import torch
from transformers import pipeline
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 初始化模型 ---
# 自动检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# a. 初始化 OWL-ViT (零样本检测器)
# 这一步不需要编译，非常简单
# device=-1 表示强制使用 CPU，如果 3070 Ti 显存紧张可以设为 -1
print("Loading OWL-ViT...")
detector = pipeline(
    model="google/owlvit-large-patch14", 
    task="zero-shot-object-detection", 
    device=0 if device.type == "cuda" else -1 
)

# b. 初始化 SAM (精细分割器)
# 需要相应的 checkpoint 文件
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth" # 替换为你的路径，推荐 ViT-H 或 ViT-L
print("Loading SAM...")
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
sam_predictor = SamPredictor(sam)


# --- 2. 加载图像 ---
image_path = "pizza.jpg" # 确保图像文件叫这个名字
image_pillow = Image.open(image_path).convert("RGB")
# 同时加载一个 OpenCV 格式用于后续画图
image_cv2 = cv2.imread(image_path)
h, w, _ = image_cv2.shape


# --- 3. 阶段一：OWL-ViT 文本定位 (无需训练) ---
# 定义你想标注的文字提示 (文本 Prompt)
# 我们这里同时寻找比萨和锡箔纸
candidate_labels = ["lasagna", "aluminum foil"] 
SCORE_THRESHOLD = 0.01 # 检测框的置信度阈值，OWL-ViT 默认偏低，可以设小一点

print(f"Processing image with text prompts: {candidate_labels}...")

# 运行检测
predictions = detector(
    image_pillow,
    candidate_labels=candidate_labels,
)

print(f"OWL-ViT found {len(predictions)} objects.")


# --- 4. 阶段二：SAM 框提示分割 (无需训练) ---
# 将结果缩放回原图大小
if len(predictions) == 0:
    print("No objects found with the given prompts.")
    exit()

# 设置 SAM 的图像特征（只需设置一次）
sam_predictor.set_image(np.array(image_pillow))

# 准备一个空白的掩码图
combined_mask = np.zeros((h, w), dtype=np.uint8)

# 遍历 OWL-ViT 找到的每一个框，让 SAM 抠图
for prediction in predictions:
    score = prediction["score"]
    if score < SCORE_THRESHOLD:
        continue
        
    label = prediction["label"]
    # 获取 Box 坐标 [x1, y1, x2, y2]
    box = prediction["box"]
    input_box = np.array([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
    
    # 使用 Box 提示传给 SAM
    masks, _, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :], # 增加一个维度
        multimask_output=False, # 只需要一个最佳掩码
    )
    
    # 将新的掩码“融合”到总掩码中
    # 这里我们简单地用逻辑或 (OR) 操作
    combined_mask = np.logical_or(combined_mask, masks[0]).astype(np.uint8) * 255


# --- 5. 提取轮廓并绘图 ---
result_img = image_cv2.copy()
# 在原图上画绿色高精度轮廓
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 500: # 过滤掉过小的杂点
        cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 4) # 绿色高精度轮廓

# --- 6. 保存结果 ---
output_path = "owlvit_sam_combined_result.jpg"
cv2.imwrite(output_path, result_img)
# 可选：也保存一份掩码图来看看
cv2.imwrite("owlvit_sam_mask_debug.png", combined_mask)

print(f"Finished. Results saved to '{output_path}'.")