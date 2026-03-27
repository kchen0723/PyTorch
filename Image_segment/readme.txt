NVILA (VLM)	晶圆级缺陷地图分类	准确率 98.4%	NVIDIA官方技术博客 + SPIE论文
DINOv2 (VFM)	晶粒级缺陷检测（光学/电子束图像）	分类精度 98.51%	NVIDIA Technical Blog
Cosmos Reason (VLM)	晶圆地图缺陷分类	微调后准确率 >96%	NVIDIA Technical Blog
SegFormer	IC芯片图像分割	mIoU 94.9%	Hugging Face模型仓库
ViT-VAE-GAN混合框架	晶圆缺陷检测	准确率提升 12.8%，误报率降低 26%	HAL学术论文


你作为一个专业的图形图像处理工程师，现在要求识别出图片中的物体并用红色把物体轮廓标注出来。比如说输入是pizza.jpg,输出要求是result.jpg. 注意图中的千层饼可以在任何位置，可以是任意形状。
用python代码来实现这个需求。
方案1：(detect_contours_perfect.py)
    A：OWL—Vit文本目标定位 ，直接根据提示词找目标
    B：用SAMsegment anything直接抠图
    C：直接描边渲染
方案2：监督学习(seggpt_contours.py)
    A：收集各种带裂纹，指纹等缺陷的图片，人工标注出来，然后用yolov8-seg或者Mask R-cnn。
    B：期间可以用SAM来辅助标注，但仅仅是辅助工具
    C：产品线上最终跑的是Yolo模型
方案3：无监督学习（unsupervised_anomaly_detection.py）
    A：先收集1000张完美的硅片，用Resnet训练完美硅片的纹理特征，并建议好完美硅片的基准线。
    B：只要测量的硅片出现模型没有见过的特征，模型立即报异常Anomaly并生成热力图Heatmap
    C：设定一个阈值，用cv2.findcontours标注出位置，用approxploydp来平滑清晰。
方案4：无监督学习（anomaly_detection_and_classification.py）
    A：先收集1000张完美的硅片，用Resnet训练完美硅片的纹理特征，并建议好完美硅片的基准线。
    B：只要测量的硅片出现模型没有见过的特征，模型立即报异常Anomaly并生成热力图Heatmap
    C：设定一个阈值，用cv2.findcontours标注出位置，用approxploydp来平滑清晰。
    D：把红色轮廓向外扩20个象素，并把ROI单独剪成一张小图
    E：用CLIP模型来识别这个ROI是哪种缺陷。
    F：最后把缺陷用红色标注出来，并把识别出的缺陷打上去。
方案5：无监督学习（anomaly_detection_and_classification.py）硅片有不同的规格
    A：先收集1000张完美的硅片，用Resnet训练完美硅片的纹理特征，并建议好完美硅片的基准线。
       不同的规格的硅片有不同的配方，将不同的配方生成不同的文件，如Wafer_Type_A_Memory.npy
    B：只要测量的硅片出现模型没有见过的特征，模型立即报异常Anomaly并生成热力图Heatmap。注意内存中只加载一个ResNet模型，然后在加载把此配方加载到基准线中。
    C：设定一个阈值，用cv2.findcontours标注出位置，用approxploydp来平滑清晰。
    D：把红色轮廓向外扩20个象素，并把ROI单独剪成一张小图
    E：用CLIP模型来识别这个ROI是哪种缺陷。
    F：最后把缺陷用红色标注出来，并把识别出的缺陷打上去。
方案6：无监督学习Dino + SAM(dinov2_segmentation.py)
    A：用DINO将图片中的物体识别出来
    B：再用SAM切割出来
方案7：无监督学习Dino + SAM + Clip(dinov2_segmentation_clip.py)
    A：用DINO将图片中的物体识别出来
    B：再用SAM切割出来
    C：最后用openai/clip-vit-base-patch32把切割出的图片识别出来。
方案8：结合无监督学习Dino + SAM + Clip（ultimate_wafer_inspection.py）
    A：先认出完美的硅片，用ResNet记忆库算法。注意不同规格的硅片有不同的配方。
    A：用DINO将图片中的物体识别出来。正常硅片的纹理则直接无视，这样不用处理那么多数据的
    B：再用SAM切割出来
    C：最后用openai/clip-vit-base-patch32把切割出的图片识别出来。
方案9：结合无监督学习Dino + SAM + Clip（ultimate_wafer_inspection.py）
    A：先认出完美的硅片，用ResNet记忆库算法。注意不同规格的硅片有不同的配方。
    A：用DINO将图片中的物体识别出来。正常硅片的纹理则直接无视，这样不用处理那么多数据的
    B：再用SAM切割出来
    C：最后用openai/clip-vit-base-patch32把切割出的图片识别出来。
    D：如果识别不了，则用线性探测。将前一步的CLIP生成数学特征向量embeddings
    E：拿这个数学特征向量，直接用scikit-learn中的logistic regression来进行分类训练即可。
方案10：视觉提示模型（seggpt_contours.py）
    A：用SegGpt或者Painter之类的大模型，给他输入一张标注好的缺陷图，然后模型会在待测试的硅片上找出这个类似的缺陷来。

