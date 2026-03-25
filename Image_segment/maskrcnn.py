import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import transforms as T

# 1. 加载预训练的 Mask R-CNN 模型
# 我们使用基于 ResNet-50 的模型，并加载权重。
def get_segmentation_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # 设置为推理模式
    # 将模型移动到 GPU (如果可用)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    return model, device

# 2. 图像预处理
# 将图像转换为 Tensor 格式
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img)
    return img_tensor, img

# COCO 数据集类别名称，用于参考
COCO_CLASS_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 3. 推理与轮廓绘制主函数
def segment_and_draw_contours(image_path, output_path):
    model, device = get_segmentation_model()
    img_tensor, original_pil_image = preprocess_image(image_path)
    
    # 转换为 OpenCV 格式用于绘制
    img_cv2 = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)

    # 模型推理
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])
    
    # 获取推理结果
    pred_score = prediction[0]['scores'].cpu().numpy()
    pred_masks = (prediction[0]['masks'] > 0.5).squeeze().cpu().numpy() # 二值化掩码
    pred_class = prediction[0]['labels'].cpu().numpy()
    
    # 筛选置信度高的检测结果
    confidence_threshold = 0.4  # 置信度阈值
    pred_t = [pred_score.tolist().index(x) for x in pred_score if x > confidence_threshold]
    
    # 如果没有检测到物体，直接返回
    if not pred_t:
        print("未检测到置信度高于 0.8 的物体。")
        cv2.imwrite(output_path, img_cv2)
        return

    # 4. 遍历检测到的物体，提取掩码并绘制轮廓
    # 轮廓颜色 (红色)
    outline_color = (0, 0, 255) # BGR
    # 轮廓粗细
    thickness = 2 

    # 为了更好的效果，我们可以根据检测类别来进行选择性绘制
    # 在标准模型中，千层面通常会被预测为：56=banana (香蕉), 59=pizza (披萨), 61=cake (蛋糕)
    # 锡箔纸可能会被预测为：52=bowl (碗) 或 28=backpack (背包，因反射)
    
    # 目标类别，仅作演示 (这取决于具体的模型和预测结果)
    target_classes = [28, 52, 59, 61] 
    
    for i in pred_t:
        label = pred_class[i]
        # if label not in target_classes:
        #    continue # 跳过不是目标的物体 (如果模型对类别判断不准确，这里需要灵活调整)
            
        print(f"检测到物体: Label={label} ({COCO_CLASS_NAMES[label]}), 置信度={pred_score[i]:.2f}")
        
        # 4a. 获取对应物体的二值掩码 (0 或 1)
        # 获取掩码 (维度是 [H, W])
        mask = pred_masks[i]
        
        # 将掩码转换为 OpenCV 兼容的 uint8 格式 (0 或 255)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 4b. 使用 OpenCV 查找轮廓 (findContours)
        # cv2.RETR_EXTERNAL: 只获取最外层轮廓
        # cv2.CHAIN_APPROX_SIMPLE: 压缩水平、垂直和对角线部分，只保留端点
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4c. 绘制轮廓 (drawContours)
        # 在原图 img_cv2 上，将 contours 画出来
        # -1: 绘制所有找到的轮廓
        cv2.drawContours(img_cv2, contours, -1, outline_color, thickness)
        
    # 5. 保存结果
    cv2.imwrite(output_path, img_cv2)
    print(f"结果已保存至: {output_path}")

# --- 运行演示 ---
if __name__ == "__main__":
    # 输入图片路径
    input_image = "pizza.jpg" # 请确保图片放在脚本同一目录中，或者填写真实路径
    output_image = "output_contours.jpg"
    
    # 创建一个测试用的图片，模拟用户提供的图片
    # 如果你本地有图片，这步可以跳过。
    # 这里通过 cv2 创建一个带有模拟物体的图片作为演示用。
    # 真实的实现代码会直接读取 your_image.jpg 
    
    # segment_and_draw_contours 将直接读取 your_image.jpg 
    try:
        segment_and_draw_contours(input_image, output_image)
    except FileNotFoundError:
        print(f"文件未找到: {input_image}。请提供有效的输入图片路径。")