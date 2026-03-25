import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

def get_smooth_contours(mask):
    """从 Mask 提取平滑、最外层的红色轮廓。"""
    # 1. 将 Mask 转换为 OpenCV 可用的 8位单通道图像
    mask_8bit = (mask * 255).astype(np.uint8)
    
    # 2. 形态学清理（关键：闭运算 - 填充内部小洞，清理杂质）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_8bit = cv2.morphologyEx(mask_8bit, cv2.MORPH_CLOSE, kernel)

    # 3. 寻找轮廓
    # RETR_EXTERNAL 保证只提取最外层边界
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    # =============== 配置区域 ===============
    # 请确保 IMAGE_PATH 和 CHECKPOINT_PATH 的路径是正确的
    IMAGE_PATH = "pizza.jpg"  # 你的原始千层面图片路径
    CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"  # 下载的模型权重路径
    MODEL_TYPE = "vit_h"
    # 使用 GPU 会快很多，如果只有 CPU，速度会较慢
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备: {DEVICE}")
    # =======================================

    # 1. 加载图片
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        print("错误：无法读取图片，请检查路径。")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # SAM 需要 RGB 格式
    h, w, _ = image_bgr.shape

    # 2. 初始化 SAM 预测器
    print("正在加载 SAM 模型并加载权重... (这可能需要一分钟)")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    
    # 3. 设置当前预测图片
    predictor.set_image(image_rgb)

    # --- 关键步骤：通过坐标点（Prompting）人为引导 AI ---
    # 我们根据图片的大致比例，设置两个点：
    # 点1: 大致位于千层面的中心 [w*0.5, h*0.6]
    # 点2: 大致位于左上角锡箔纸的边缘 [w*0.2, h*0.2]
    # 这样 AI 就知道你要分割的是这两个区域及其合并体
    input_points = np.array([[w//2, int(h*0.6)], [int(w*0.2), int(h*0.2)]])
    input_labels = np.array([1, 1]) # 1 代表“这是我要的物体”

    # 4. 让 AI 生成掩膜 (全图分割)
    print("正在进行交互式分割... (这非常快)")
    # predictor.predict 可以通过点引导生成 Mask
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True, # 输出不同层级的分割
    )
    print(f"检测到 {len(masks)} 个潜在物体区域层级。")

    if len(masks) == 0:
        print("未检测到物体。")
        return

    # 5. 绘图与筛选逻辑（复现我那个效果的关键）
    # 我们通常取 scores 最高的两个不同层级的 mask
    # 第一个（masks[0]）通常是较小的那个（比如只分割千层面）
    # 第二个（masks[1]）通常是包含容器的整体（比如千层面+锡箔纸）
    
    # 为了复现效果，我们将两个 Mask 合并（因为我的原图是将两个轮廓都画出来的）
    # 如果你只想要千层面，只需用其中一个 Mask 即可
    final_masks_union = np.zeros_like(masks[0])
    for i in range(2): # 取两个层级
        final_masks_union = cv2.bitwise_or(final_masks_union, masks[i])

    # 6. 提取平滑、清晰的轮廓
    print("正在清理轮廓线并绘制...")
    cnts_smooth = get_smooth_contours(final_masks_union)
    
    # 7. 绘制轮廓
    # 颜色在 OpenCV 中是 BGR 格式
    line_color = (0, 0, 255) # 千层面用纯红 (为了醒目)
    thickness = 6 # 粗线，防止杂质

    # A: 在原图上绘制轮廓
    # color 在 OpenCV 中是 BGR 格式
    final_image = image_bgr.copy()
    # 使用 cv2.LINE_AA 开启抗锯齿，使线条边缘平滑
    cv2.drawContours(final_image, cnts_smooth, -1, line_color, thickness, cv2.LINE_AA)

    # 8. 保存并显示结果
    output_path = "lasagna_perfect_outline_guided.jpg"
    cv2.imwrite(output_path, final_image)
    print(f"结果已保存至: {output_path}")

    # (可选) 使用 Matplotlib 显示
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title("清晰平滑的引导分割轮廓（复现效果图）")
    # plt.show()

if __name__ == "__main__":
    main()