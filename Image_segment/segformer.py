import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# 1. 初始化模型和处理器
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def segment_and_draw(image_path, output_path):
    # 2. 读取图片
    image = Image.open(image_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    w, h = image.size

    # 3. 推理获取分割图
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 将结果缩放到原图大小
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    seg_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    # 4. 定义需要标注的类别 ID (以 ADE20K 为例)
    # 常用 ID: 93(food/pizza), 116(tray/foil-like), 14(curtain/foil-like)
    # 注意：具体 ID 取决于模型训练集，此处仅作演示
    target_ids = {
        "Pizza/Food": 93, 
        "Foil/Tray": 116
    }
    colors = {"Pizza/Food": (0, 0, 255), "Foil/Tray": (0, 255, 0)} # 红色和绿色

    for name, class_id in target_ids.items():
        # 创建该类别的二值掩膜
        mask = np.where(seg_map == class_id, 255, 0).astype(np.uint8)
        
        # 5. 提取并绘制轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_cv, contours, -1, colors[name], 2)
        
        # 在轮廓上方写字
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, _, _ = cv2.boundingRect(c)
            cv2.putText(img_cv, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[name], 2)

    # 6. 保存结果到硬盘
    cv2.imwrite(output_path, img_cv)
    print(f"标注完成，图片已保存至: {output_path}")

# 执行
segment_and_draw("pizza.jpg", "annotated_pizza.jpg")