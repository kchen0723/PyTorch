import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- 1. 模型初始化 (ViT-L + FP16) ---
model_type = "vit_h"
checkpoint = "sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)

# 调整参数以减少细碎程度
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=24,           # 降低采样点，减少褶皱干扰
    pred_iou_thresh=2.8,          # 稍微放宽，允许捕获更多铝箔纸区域
    stability_score_thresh=0.9,
    min_mask_region_area=200      # 过滤微小杂点
)

# --- 2. 处理图片 ---
image = cv2.imread("pizza.jpg")
h, w = image.shape[:2]
img_center = (w // 2, h // 2)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("正在全自动分析图片...")
masks = mask_generator.generate(image_rgb)

# --- 3. 智能逻辑筛选 ---
# 提取掩码特征
scored_masks = []
for m in masks:
    seg = m['segmentation']
    area = np.sum(seg)
    x, y, bw, bh = m['bbox']
    center = (x + bw//2, y + bh//2)
    dist = np.sqrt((center[0]-img_center[0])**2 + (center[1]-img_center[1])**2)
    
    scored_masks.append({
        'mask': seg,
        'area': area,
        'dist': dist,
        'bbox': m['bbox']
    })

# A. 锁定比萨：距离中心近、面积适中、非全图背景
# 我们选取距离中心 15% 范围内最大的掩码
pizza_candidates = [m for m in scored_masks if m['dist'] < (w * 0.15) and m['area'] < (h*w*0.6)]
pizza_candidates.sort(key=lambda x: x['area'], reverse=True)
pizza_mask = pizza_candidates[0]['mask'] if pizza_candidates else None

# B. 寻找锡箔纸：合并所有非比萨的大型碎片
foil_union = np.zeros((h, w), dtype=np.uint8)
for m in scored_masks:
    # 如果该掩码不是比萨，且面积足够大（大于全图1%），则视为铝箔纸的一部分
    if m['area'] > (h * w * 0.01):
        # 排除掉已经是比萨的区域
        if pizza_mask is not None:
            if np.logical_and(m['mask'], pizza_mask).sum() / m['area'] > 0.8:
                continue
        foil_union[m['mask']] = 255

# --- 4. 关键：轮廓平滑与填补 ---
# 使用闭运算连接碎裂的铝箔纸
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
foil_closed = cv2.morphologyEx(foil_union, cv2.MORPH_CLOSE, kernel)

# 寻找合并后最大的外部轮廓
contours_f, _ = cv2.findContours(foil_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
final_foil_contour = max(contours_f, key=cv2.contourArea) if contours_f else None

# --- 5. 绘制结果 ---
result = image.copy()
# 画锡箔纸 (蓝色)
if final_foil_contour is not None:
    cv2.drawContours(result, [final_foil_contour], -1, (255, 0, 0), 3)
    cv2.putText(result, "Foil", tuple(final_foil_contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# 画比萨 (绿色)
if pizza_mask is not None:
    contours_p, _ = cv2.findContours(pizza_mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_p:
        best_p = max(contours_p, key=cv2.contourArea)
        cv2.drawContours(result, [best_p], -1, (0, 255, 0), 3)
        cv2.putText(result, "Pizza", tuple(best_p[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imwrite("sam_auto_final.jpg", result)
print("处理完成，请检查 sam_auto_final.jpg")