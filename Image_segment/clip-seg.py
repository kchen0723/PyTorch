import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import cv2
import numpy as np

# --- 1. 初始化模型与配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

# 加载模型和处理器
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.to(device)
model.eval()

# --- 2. 参数设置 ---
IMAGE_PATH = "pizza.jpg"  # 你的图片路径
PROMPTS = ["pizza", "aluminum foil"]  # 你想识别的多个物体
THRESHOLD = 0.4  # 敏感度阈值 (0-1)，调低会选出更多区域

# --- 3. 图像加载 ---
image_pillow = Image.open(IMAGE_PATH).convert("RGB")
image_cv2 = cv2.imread(IMAGE_PATH)
h, w, _ = image_cv2.shape

# --- 4. 预处理与推理 ---
print(f"正在识别: {PROMPTS}...")

# 关键修复：添加 padding=True 解决长度不一导致的报错
inputs = processor(
    text=PROMPTS, 
    images=[image_pillow] * len(PROMPTS), 
    padding=True, 
    return_tensors="pt"
)

# 移至 GPU
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

# logits 维度为 [num_prompts, 352, 352]
logits = outputs.logits
if len(PROMPTS) == 1:
    logits = logits.unsqueeze(0)

# --- 5. 后处理：缩放与合并掩码 ---
# 将结果缩放回原图大小
preds = torch.sigmoid(logits) # 转为概率
preds_resized = torch.nn.functional.interpolate(
    preds.unsqueeze(1), 
    size=(h, w), 
    mode="bilinear", 
    align_corners=False
).squeeze(1)

# 将所有 Prompts 的结果合并 (取最大值)
combined_mask_tensor = torch.max(preds_resized, dim=0)[0]
binary_mask = (combined_mask_tensor.cpu().numpy() > THRESHOLD).astype(np.uint8) * 255

# --- 6. 提取轮廓并绘图 ---
result_img = image_cv2.copy()
# 颜色方案：比萨用绿色，锡箔纸用蓝色 (这里为了演示方便，统一合并后提取轮廓)
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 500:  # 过滤掉过小的杂点
        # 画出轮廓
        cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 3)
        # 获取轮廓位置标注文字
        x, y, _, _ = cv2.boundingRect(cnt)
        cv2.putText(result_img, "Detected", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# --- 7. 保存结果 ---
cv2.imwrite("clipseg_combined_result.jpg", result_img)
cv2.imwrite("clipseg_mask_debug.png", binary_mask) # 调试用的黑白掩码图

print("处理完成！结果已保存为 'clipseg_combined_result.jpg'")