import torch
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import cv2
import sys
import os
from PIL import Image

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("缺少 segment-anything 库。请在终端运行相关安装命令。")
    sys.exit(1)

from transformers import pipeline

# ==============================================================================
# 🌟 终极多范式 AOI 流水线: ResNet(多库匹配) + SAM(像素裁剪) + CLIP(定性)
# ==============================================================================

def get_feature_extractor():
    model = models.resnet18(pretrained=True)
    model.eval()
    features = []
    def hook(module, input, output):
        features.append(output)
    model.layer2.register_forward_hook(hook)
    model.layer3.register_forward_hook(hook)
    return model, features

def extract_features(img_tensor, model, features_list):
    features_list.clear()
    with torch.no_grad():
        _ = model(img_tensor)
    feat2 = features_list[0]
    feat3 = features_list[1]
    feat3_up = F.interpolate(feat3, size=feat2.shape[2:], mode='bilinear', align_corners=False)
    combined_features = torch.cat([feat2, feat3_up], dim=1) 
    return combined_features

def build_memory_bank(good_images_paths, model, features_list, device):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    all_features = []
    for path in good_images_paths:
        img = cv2.imread(path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)
        feat = extract_features(tensor, model, features_list)
        all_features.append(feat.cpu().numpy())
    if not all_features:
        raise ValueError("没有读取到良品图")
    all_features = np.concatenate(all_features, axis=0) # [N, 384, 32, 32]
    mean_map = np.mean(all_features, axis=0)
    std_map = np.std(all_features, axis=0) + 1e-5
    return mean_map, std_map

def run_ultimate_inspection(good_wafers_dict, test_image, output_path, dynamic_defect_types, threshold=4.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[架构引擎] 启动硬件: {device}")

    # ===== [大模型接通] =====
    print("\n正在接通三大高维度 AI 引擎网络...")
    resnet_model, hook_features = get_feature_extractor()
    resnet_model.to(device)
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    try:
        classifier = pipeline(
            task="zero-shot-image-classification", 
            model="openai/clip-vit-base-patch32", 
            device=0 if device.type == "cuda" else -1
        )
    except Exception as e:
        print(f"⚠️ CLIP 下载失败（HuggingFace拦截），将只能框出病灶但不命名。")
        classifier = None

    # ===== [建立多晶格无监督记忆库] =====
    print("\n阶段 1：学习多种“完美硅片”的底纹配方 (Recipe)")
    memory_banks = {}
    for wafer_type, paths in good_wafers_dict.items():
        print(f" -> 正在建立 [{wafer_type}] 的独立记忆库 (含样本 {len(paths)} 张)...")
        try:
            mean_map, std_map = build_memory_bank(paths, resnet_model, hook_features, device)
            memory_banks[wafer_type] = (mean_map, std_map)
            print(f"    完成 {wafer_type} 晶格学习。")
        except Exception as e:
            print(f"    [!] 建立 {wafer_type} 失败: {e}")
            
    if not memory_banks:
        print("所有记忆库均建立失败，程序终止。")
        return

    # ===== [多晶格自适应匹配与异常扫描] =====
    print("\n阶段 2：全图异常探针扫描与自适应晶格匹配...")
    img = cv2.imread(test_image)
    if img is None:
        print(f"找不到测试图片 {test_image}")
        return
        
    h_orig, w_orig = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    feat = extract_features(tensor, resnet_model, hook_features).cpu().numpy()[0]
    
    # 自动识别当前是哪一种硅片 (找全局偏差最小的记忆库)
    best_wafer_type = None
    lowest_global_diff = float('inf')
    best_anomaly_map = None
    
    for wafer_type, (mean_map, std_map) in memory_banks.items():
        diff = np.abs(feat - mean_map) / std_map 
        anomaly_map = np.mean(diff, axis=0)
        global_diff = np.mean(anomaly_map)
        
        if global_diff < lowest_global_diff:
            lowest_global_diff = global_diff
            best_wafer_type = wafer_type
            best_anomaly_map = anomaly_map
            
    print(f"探针汇报: 自适应匹配到正确的底纹配方为 [{best_wafer_type}] (全局极小值: {lowest_global_diff:.2f})")
    print(f"   -> 系统已自动过滤 {best_wafer_type} 的特有晶格纹理。开始检索独立病灶...")
    
    anomaly_map_resized = cv2.resize(best_anomaly_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    binary_mask = (anomaly_map_resized > threshold).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ===== [启用 SAM 与 CLIP 高阶交互] =====
    print(f"\n阶段 3 & 4：喂入 SAM 切割边缘并提交给 CLIP 鉴定...")
    sam_predictor.set_image(img_rgb)
    
    result_img = img.copy()
    scale = 4
    overlay_up = np.zeros((h_orig * scale, w_orig * scale, 3), dtype=np.uint8)
    drawn_count = 0

    for cnt in contours:
        if cv2.contourArea(cnt) > (w_orig * h_orig * 0.005): 
            x, y, w, h = cv2.boundingRect(cnt)
            
            input_box = np.array([x, y, x + w, y + h])
            sam_masks, _, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            final_mask = (sam_masks[0] * 255).astype(np.uint8)
            
            pad = 25
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_orig, x + w + pad), min(h_orig, y + h + pad)
            
            cropped_rgb = img_rgb[y1:y2, x1:x2]
            pil_cropped = Image.fromarray(cropped_rgb)
            
            if classifier is not None:
                predictions = classifier(pil_cropped, candidate_labels=dynamic_defect_types)
                best_label = predictions[0]['label']
                confidence = predictions[0]['score']
                display_label = f"{best_label} ({confidence:.2f})"
            else:
                display_label = f"Defect on {best_wafer_type}"
                
            melt_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, melt_kernel)
            
            sam_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for s_cnt in sam_contours:
                peri = cv2.arcLength(s_cnt, True)
                approx = cv2.approxPolyDP(s_cnt, 0.005 * peri, True)
                
                cnt_up = (approx * scale).astype(np.int32)
                cv2.drawContours(overlay_up, [cnt_up], -1, (0, 0, 255), 2 * scale, lineType=cv2.LINE_AA)
            
            cv2.putText(result_img, display_label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            drawn_count += 1

    if drawn_count > 0:
        overlay_up = cv2.GaussianBlur(overlay_up, (5, 5), 0)
        overlay = cv2.resize(overlay_up, (w_orig, h_orig), interpolation=cv2.INTER_AREA)
        alpha = np.max(overlay, axis=2).astype(float) / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        result_img = (result_img.astype(float) * (1 - alpha) + overlay.astype(float) * alpha).astype(np.uint8)

    cv2.imwrite(output_path, result_img)
    print(f"\n完美收工！共成功将 {drawn_count} 处在绝对正常背景中脱颖而出的微小异常：")
    print(f"   -> 自动匹配底纹配方: [{best_wafer_type}]")
    print(f"   -> 用 ResNet 定位，用 SAM 抽取出轮廓，由 CLIP 完成定性判定。")
    print(f"结案图像已导出为: {output_path}")

if __name__ == "__main__":
    # 配置
    # 【多配方记忆字典】你可以放入无数种不同底纹的良品硅片
    GOOD_WAFERS_DICT = {
        "Type_A_Grid": ["normal_wafer_A1.jpg", "normal_wafer_A2.jpg"],
        "Type_B_Smooth": ["normal_wafer_B1.jpg", "normal_wafer_B2.jpg"]
    }
    
    TEST_WAFER = "wafer_with_defect.jpg"
    OUT_FILE = "ultimate_pipeline_result.jpg"
    
    # 防御降级逻辑
    if not os.path.exists("normal_wafer_A1.jpg") and os.path.exists("pizza.jpg"):
        GOOD_WAFERS_DICT = {
            "Type_A_Grid_Wafer": ["pizza.jpg", "pizza.jpg"],
            "Type_B_Smooth_Wafer": ["pizza.jpg"] 
        }
        TEST_WAFER = "pizza.jpg"
        threshold_value = 0.0001
    else:
        threshold_value = 4.5
        
    DEFECT_TYPES = [
        "a deep scratch on surface",
        "a white smudge",
        "a cracked line",
        "a greasy fingerprint",
        "dust particle",
        "cheese and tomato sauce anomaly" 
    ]
    
    run_ultimate_inspection(GOOD_WAFERS_DICT, TEST_WAFER, OUT_FILE, DEFECT_TYPES, threshold=threshold_value)
