import torch
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import cv2
import os

# ==============================================================================
# 🌟 方案B：无监督异常检测 (Unsupervised Anomaly Detection) 轮廓提取脚本
# ==============================================================================
# 架构原理 (基于极简版的 PaDiM 算法)：
# 1. 阶段一 (训练)：输入只有“完美无瑕的良品图”(Good Images)。
#    使用预训练的 ResNet 提取中高层特征，建立一个描述“正常外观”的高斯密度函数分布 (Memory Bank)。
# 2. 阶段二 (推理)：输入“可能有缺陷的待测图”(Test Image)。
#    提取特征并与正常的高斯分布计算马氏距离，异常区域特征偏移巨大，从而生成“异常热力图”。
# 3. 阶段三 (后处理)：对热力图进行阈值分割，找出超过阈值的缺陷区域（如指纹、裂纹），画出红色轮廓。

# 脚本的工作原理：
# 阶段一（无监督学习 / 建立记忆库）： 你不需要标注任何缺陷！只需要提供几十张“完美无瑕的良品硅片图片”。脚本会使用预训练的 ResNet 提取它们的中高层纹理特征，并通过统计学计算建立一个“正常硅片应该长什么样”的基准线（高斯分布）。
# 阶段二（待测图比对计算距离）： 输入一张带指纹或微裂纹的硅片图。脚本同样提取它的特征，并与刚才建立的“良品记忆库”进行比对。哪怕原本极其难以察觉的微小指纹，在这里也会因为特征偏离了正常分布而被放大，计算出巨大的偏离得分，形成一张异常热力图（Heatmap）。
# 阶段三（红色轮廓提取）： 通过设定一个阈值（过滤掉正常的微小光影波动），将热力图二值化。接着调用我们之前一贯的专业画法：使用 cv2.findContours 找到病灶边缘，应用 approxPolyDP 使边缘平滑清晰，最后用显眼的红色 

# (0, 0, 255)
#  在原图上画出线宽为 2 的抗锯齿轮廓，并在图上打上 Defect (Anomaly) 的标签。
# ==============================================================================

def get_feature_extractor():
    """获取预训练特征提取器 (ResNet18) 和特征挂钩"""
    model = models.resnet18(pretrained=True)
    model.eval() # 必须是评估模式
    
    features = []
    # 我们截取它的中层网络 (layer2 和 layer3)
    # 因为浅层往往是基础边缘，深层是对狗和猫的语义，中层更能捕获异常的工业纹理和局部结构
    def hook(module, input, output):
        features.append(output)
        
    model.layer2.register_forward_hook(hook)
    model.layer3.register_forward_hook(hook)
    return model, features

def extract_features(img_tensor, model, features_list):
    """提取对齐拼接后的中层特征张量"""
    features_list.clear() # 清除上次推断留下的特征
    with torch.no_grad():
        _ = model(img_tensor)
    
    feat2 = features_list[0] # 大小如 [B, 128, 32, 32]
    feat3 = features_list[1] # 大小如 [B, 256, 16, 16]
    
    # 对齐尺寸: 将 feat3 上采样到 feat2 的尺寸
    feat3_up = F.interpolate(feat3, size=feat2.shape[2:], mode='bilinear', align_corners=False)
    
    # 在通道维度拼接，形成强大的融合特征描述符 [B, 384, 32, 32]
    combined_features = torch.cat([feat2, feat3_up], dim=1) 
    return combined_features

def build_memory_bank(good_images_paths, model, features_list, device):
    """训练期：提取良品特征，计算正常分布的均值和标准差"""
    print(f"⚙️ 正在学习 {len(good_images_paths)} 张良品硅片的正常纹理特征...")
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)), # 规范化尺寸
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
        raise ValueError("❌ 没有在此路径下成功读取到任何良品图")

    all_features = np.concatenate(all_features, axis=0) # [N, 384, 32, 32]
    
    # 我们在每个空间像素位置上，计算所有良品样本在通道方向的均值和标准差
    mean_map = np.mean(all_features, axis=0) # [384, 32, 32]
    std_map = np.std(all_features, axis=0) + 1e-5 # [384, 32, 32]，+ epsilon 防止除0
    
    return mean_map, std_map

def detect_anomaly_and_draw_contours(test_img_path, mean_map, std_map, model, features_list, device, threshold=3.5):
    """推理期：根据偏离度生成热力图，并在原图上刻画红色轮廓"""
    img = cv2.imread(test_img_path)
    if img is None:
        print(f"❌ 找不到需要预测的测试图: {test_img_path}")
        return None, None
    
    h_orig, w_orig = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 保持与良品相同的预处理规范
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    # 提取测试图的特征
    feat = extract_features(tensor, model, features_list).cpu().numpy()[0] # [384, 32, 32]
    
    # 计算差异得分 (类似于对角化的马氏距离)
    # 即：测试像素特征偏离了正常特征均值多少个标准差
    diff = np.abs(feat - mean_map) / std_map 
    anomaly_map = np.mean(diff, axis=0) # 平均下各个维度的偏离度，合并为一张热力图 [32, 32]
    
    # 将热力图平滑拉伸回原图的原本尺寸
    anamoly_map_resized = cv2.resize(anomaly_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    # 报告分数范围，方便工程师动态调整 threshold
    print(f"📊 得分分布 -> 最小值(正常): {anamoly_map_resized.min():.2f}, 最大值(缺陷): {anamoly_map_resized.max():.2f}")
    
    # 1. 阈值分割：找出异常得分高于设定阈值的地方
    binary_mask = (anamoly_map_resized > threshold).astype(np.uint8) * 255
    
    # 2. 图像学清理：抹平噪点和孔洞
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel) # 侵蚀孤立点
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel) # 填平破口

    result_img = img.copy()
    
    # 3. 提取轮廓并在缺陷外围画大红色描边
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    defect_count = 0
    for cnt in contours:
        # 跳过过于细小的微尘噪点
        if cv2.contourArea(cnt) > 200: 
            # 平滑缺陷的边缘，使其符合之前要求的专业描边标准
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.005 * peri, True)
            
            # 绘制 2 像素级的红色轮廓
            cv2.drawContours(result_img, [approx], -1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            
            x, y, _, _ = cv2.boundingRect(approx)
            cv2.putText(result_img, "Defect (Anomaly)", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            defect_count += 1
            
    print(f"🚨 共画出并标记了 {defect_count} 处异常缺陷。")
    return result_img, anamoly_map_resized


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在配置无监督计算设备: {device}...")
    
    # ---------------- 业务路径配置 ----------------
    # 真实场景中，你会把 100 张正常硅片的照片丢在 normal_wafers 目录下
    good_images_paths = ["normal_wafer_1.jpg", "normal_wafer_2.jpg", "normal_wafer_3.jpg"] 
    test_image_path = "wafer_with_fingerprint_or_crack.jpg" # 带有待测缺陷的图片
    threshold_value = 4.5 # 表示偏离平均分布 4.5 倍以上的区域判定为异常病灶
    # ----------------------------------------------
    
    # 演示防御逻辑：如果硬盘里根本没有这几张硅片图片文件，为了让你能跑通代码，它会调用现成的 pizza.jpg 充数
    if not os.path.exists(good_images_paths[0]) and os.path.exists("pizza.jpg"):
        print("\n⚠️ 警告: 未找到工业硅片图片，将借用现有的 pizza.jpg 供代码调试畅通。")
        good_images_paths = ["pizza.jpg"]
        test_image_path = "pizza.jpg"
        # 自己和自己比差异是 0，所以把阈值降极低来强行触发检测演示
        threshold_value = 0.0001
        
    print("\n--- 阶段一：建立无监督学习记忆库 ---")
    model, hook_features = get_feature_extractor()
    model.to(device)
    mean_map, std_map = build_memory_bank(good_images_paths, model, hook_features, device)
    print("✅ 记忆库状态（正常状态基准线）构建完成。")
    
    print("\n--- 阶段二：计算异常与轮廓抽取 ---")
    result_img, heatmap = detect_anomaly_and_draw_contours(
        test_image_path, mean_map, std_map, model, hook_features, device, threshold=threshold_value
    )
    
    if result_img is not None:
        cv2.imwrite("anomaly_contours_result.jpg", result_img)
        print("📁 缺陷特征已用红框标出并自动保存为: anomaly_contours_result.jpg")
        
        # 附带保存热力图（工业界排错最喜欢看的图：颜色越红说明特征偏离越离谱，指纹/裂纹的位置会发红）
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        cv2.imwrite("anomaly_heatmap.jpg", heatmap_color)
        print("🔥 探针误差热力图已附加保存为: anomaly_heatmap.jpg")
