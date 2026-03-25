import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps
import sys
import os

# 导入 SAM 库
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("错误：未找到 segment-anything 库。请运行: pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)

def generate_masks_sam(image_path, model_path="sam_vit_b_01ec64.pth", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    第一步：使用 SAM 自动生成万物掩码 (精确区分千层面和锡箔纸)
    """
    if not os.path.exists(model_path):
        print(f"错误：没找到 SAM 模型权重文件 '{model_path}'。请先下载。")
        sys.exit(1)

    # 1a. 初始化 SAM 模型
    print(f"正在加载 SAM 模型 ({device})... 这可能需要一分钟...")
    model_type = "vit_b" # 使用 Base 模型以平衡速度和质量
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)

    # 1b. 创建全自动掩码生成器
    # 我们可以调整参数以获得更多或更少的掩码。
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,          # 每边采样的点数 (越高越细)
        pred_iou_thresh=0.88,         # 预测 IoU 阈值 (控制精度)
        stability_score_thresh=0.95, # 稳定性分数阈值 (减少噪点)
        crop_n_layers=0,             # 禁用切片 (加快速度)
        min_mask_region_area=200,    # 过滤掉过小的掩码
    )

    # 1c. 读取图片并推理
    img_cv2 = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    
    print(f"正在分析图片 '{image_path}' 的结构...")
    # 这是 SAM 的“魔法”时刻，它会识别所有物体
    masks_data = mask_generator.generate(img_rgb)
    
    # 1d. 处理生成的掩码
    print(f"SAM 识别出了 {len(masks_data)} 个潜在物体。正在提取千层面和锡箔纸...")
    
    H, W = img_rgb.shape[:2]
    final_combined_mask = np.zeros((H, W), dtype=np.uint8)

    # 模拟处理：我们需要合并千层面和锡箔纸这两个主要掩码。
    # 通常最大的掩码是锡箔纸，紧随其后的是千层面。
    # 我们按面积排序，提取前两个最大的、置信度高的物体。
    sorted_masks = sorted(masks_data, key=(lambda x: x['area']), reverse=True)
    
    # 我们通常需要合并至少前两个物体来捕捉整体结构。
    target_masks_to_combine = 2 
    for i, mask_entry in enumerate(sorted_masks):
        if i >= target_masks_to_combine:
            break
            
        # 将布尔掩码转为二值化 (0 或 255)
        m = (mask_entry['segmentation']).astype(np.uint8) * 255
        # 使用逻辑“或”合并
        final_combined_mask = cv2.bitwise_or(final_combined_mask, m)
        
    return final_combined_mask

def draw_smooth_outline_dual(image_path, mask_np, output_path):
    """
    第二步：使用 PIL 进行高质量、抗锯齿双层渲染 (复现我的效果)
    """
    img_pil = Image.open(image_path).convert("RGBA")
    # 将 numpy mask 转为 PIL Image
    mask_pil = Image.fromarray(mask_np).convert("L")

    # 1. 提取边缘 (FIND_EDGES 会为掩码的内部和外部都提取边缘，实现双层效果)
    # 这就是关键差异所在：之前的 FIND_EDGES 只作用于单层千层面，
    # 而这次 FIND_EDGES 作用于包含锡箔纸和千层面整体的掩码。
    edges = mask_pil.filter(ImageFilter.FIND_EDGES)
    
    # 2. 膨胀并模糊 (关键：实现平滑抗锯齿)
    thickness = 5  # 轮廓粗细
    smoothness = 4  # 平滑度 (越高越丝滑)
    
    # 膨胀边缘以控制粗细
    dilated_edges = edges.filter(ImageFilter.MaxFilter(thickness * 2 + 1))
    
    # 高斯模糊实现平滑 Alpha 混合
    smoothed_edges = dilated_edges.filter(ImageFilter.GaussianBlur(radius=smoothness))
    
    # 3. 创建红色层并应用Alpha通道
    # 颜色 (红色): (255, 0, 0)
    red_layer = Image.new("RGBA", img_pil.size, (255, 0, 0, 255))
    outline_layer = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    outline_layer.paste(red_layer, (0, 0), smoothed_edges)

    # 4. 复合回原图
    # PIL 会自动处理 Alpha 混合，得到完美贴合、平滑的红色边缘。
    final_img = Image.alpha_composite(img_pil, outline_layer)
    
    # 5. 保存结果 (质量设为最高)
    final_img.convert("RGB").save(output_path, quality=95)
    print(f"成功！高质量双层轮廓图片已保存至: {output_path}")

if __name__ == "__main__":
    # 输入图片文件名 (确保你的文件名叫这个)
    input_file = "pizza.jpg" 
    output_file = "result_sam_dual_outline.jpg"

    # 如果没有指定GPU，SAM会很慢，提醒一下
    if not torch.cuda.is_available():
        print("警告：未检测到 GPU (CUDA)。SAM 推理将使用 CPU，速度将非常缓慢。")

    try:
        # 第一步：全自动生成精确的万物掩码 (包含锡箔纸)
        # 这也是 AI 领域真正通用的本地实现方案
        print("正在使用 SAM (Segment Anything Model) 初始化双层轮廓提取流程...")
        auto_mask = generate_masks_sam(input_file)
        
        # 第二步：高质量平滑 Alpha 渲染
        draw_smooth_outline_dual(input_file, auto_mask, output_file)
        
    except FileNotFoundError:
        print(f"错误：没找到 {input_file}，请确认图片放在代码旁边。")
    except Exception as e:
        print(f"发生错误: {e}")