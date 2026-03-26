import torch
import torchvision.transforms as T
import cv2
import numpy as np
import sys
from PIL import Image

try:
    from sklearn.cluster import KMeans
except ImportError:
    print("❌ 缺少 scikit-learn 库。请在终端运行: pip install scikit-learn")
    sys.exit(1)

from segment_anything import sam_model_registry, SamPredictor

# ==============================================================================
# 🌟 基于 DINOv2 + SAM 的无监督像素级语义分割 (Hybrid Edge Refinement)
# ==============================================================================
# 原理升级：
# 1. 解决了 DINOv2 "14x14 Patch" 导致的边缘粗糙碎裂问题。
# 2. 我们依然完全不使用任何文字 Prompt，使用 DINOv2 进行神仙级的**纯无监督语义聚类**。
# 3. 拿到 DINOv2 的粗糙原始色块后，我们精确提取这些色块的 Bounding Box (边界框)，
# 4. 将这些边界框作为零人工介入的“视觉提示 (Prompt)”喂给 Segment Anything (SAM)！
# 5. SAM 是一把边缘手术刀，它会在 DINO 定位的框内，自动寻找并贴合真实物体的亚像素级边缘。
# 这样就完美结合了 DINOv2 的“强语义理解”和 SAM 的“高清边缘锐利度”！
# ==============================================================================

def run_dinov2_sam_segmentation(img_path, output_path, k_clusters=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"配置计算设备: {device}")

    # ================= 1. 加载所有大模型 =================
    print("正在加载双引擎流水线...")
    print(" 1/2 加载 DINOv2-Base (全局语义大脑)...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2_model.eval().to(device)
    
    print(" 2/2 加载 SAM-ViT-H (像素级边缘手术刀)...")
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    # ================= 2. 读取图片并适配 DINOv2 =================
    image_pillow = Image.open(img_path).convert("RGB")
    img_cv2 = cv2.imread(img_path)
    if img_cv2 is None: return

    w_orig, h_orig = image_pillow.size
    new_w = (w_orig // 14) * 14
    new_h = (h_orig // 14) * 14
    image_resized = image_pillow.resize((new_w, new_h), Image.LANCZOS)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(image_resized).unsqueeze(0).to(device)

    # ================= 3. DINOv2 提取语义簇 (大颗粒) =================
    print("阶段一：DINOv2 开始无监督阅读图片语义...")
    with torch.no_grad():
        features = dinov2_model.forward_features(img_tensor)
        patch_tokens = features['x_norm_patchtokens'].squeeze(0).cpu().numpy()
        
    num_patches_w = new_w // 14
    num_patches_h = new_h // 14

    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(patch_tokens)
    
    mask_2d = labels.reshape(num_patches_h, num_patches_w).astype(np.uint8)
    mask_full = cv2.resize(mask_2d, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    corners = [mask_full[0, 0], mask_full[0, -1], mask_full[-1, 0], mask_full[-1, -1]]
    bg_cluster_id = max(set(corners), key=corners.count)

    # ================= 4. SAM 精细边缘切割 =================
    print("阶段二：SAM 开始接受 DINOv2 的引导，进行像素级边缘重构...")
    sam_predictor.set_image(np.array(image_pillow))
    
    result_img = img_cv2.copy()
    
    # 引入 4x 超级采样机制，实现彻底的“液态顺滑”线框美学
    scale = 4
    overlay_up = np.zeros((h_orig * scale, w_orig * scale, 3), dtype=np.uint8)
    drawn_count = 0

    for cluster_id in range(k_clusters):
        if cluster_id == bg_cluster_id:
            continue
            
        binary_mask = (mask_full == cluster_id).astype(np.uint8) * 255
        kernel = np.ones((25, 25), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        x, y, w, h = cv2.boundingRect(binary_mask)
        if w < 20 or h < 20: continue 
        
        input_box = np.array([x, y, x + w, y + h])
        sam_masks, _, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        # SAM Mask 是真实物理边缘，有很多细小褶皱
        final_mask = (sam_masks[0] * 255).astype(np.uint8)
        
        # 使用强力形态学闭运算 (Close) 强制填平锡箔纸边缘的细小狗牙裂口
        melt_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, melt_kernel)
        
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > (w_orig * h_orig * 0.05):
                # 调大 epsilon 系数 (0.005)，强行抹平真实的物理褶皱，达到“人类手绘”的抽象圆滑感
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.005 * peri, True)
                
                # 在 4 倍放大的高分辨率画布上作画
                cnt_up = (approx * scale).astype(np.int32)
                cv2.drawContours(overlay_up, [cnt_up], -1, (0, 0, 255), 2 * scale, lineType=cv2.LINE_AA)
                
                # 缩放坐标用于文字
                x_text, y_text = approx[0][0]
                cv2.putText(result_img, f"DINOv2+SAM Sub-Object {cluster_id}", (x_text, y_text - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                drawn_count += 1

    # 4x 渲染收尾：缩放回原图以获得无与伦比的抗锯齿效果
    if drawn_count > 0:
        overlay_up = cv2.GaussianBlur(overlay_up, (5, 5), 0)
        overlay = cv2.resize(overlay_up, (w_orig, h_orig), interpolation=cv2.INTER_AREA)
        alpha = np.max(overlay, axis=2).astype(float) / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        result_img = (result_img.astype(float) * (1 - alpha) + overlay.astype(float) * alpha).astype(np.uint8)

    cv2.imwrite(output_path, result_img)
    print(f"处理完成！依靠双核大模型渲染了 {drawn_count} 组极其精准且液滑的语义级轮廓。")

if __name__ == "__main__":
    run_dinov2_sam_segmentation("pizza.jpg", "dinov2_result.jpg", k_clusters=3)
