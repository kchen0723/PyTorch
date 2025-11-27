base install:
pip install transformers
pip install transformers[torch]
pip install transformers[tf]
pip install transformers[torch, tf, audio, vision]

dependency install:
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib
pip install datasets
pip install accerate
pip install evalute

https://blog.csdn.net/qq_75211380/article/details/149243540
http://www.360doc.com/content/25/1019/22/62738899_1163323291.shtml

To downlaod GODEL model:
Install Git LFS,
apt-get update
apt-get install -y git-lfs
git lfs install

Copy model with git LFS:
git clone https://huggingface.co/microsoft/GODEL-v1_1-base-seq2seq /models/GODEL-v1_1-base-seq2seq

Now Load with transformers:
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
path = "/models/GODEL-v1_1-base-seq2seq"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSeq2SeqLM.from_pretrained(path)

=================================================================================
HuggingFace Transformers 内置 pipeline() 全任务列表
🟦 NLP（自然语言处理）任务
pipeline 名称	功能
sentiment-analysis	情感分析
text-classification	文本分类（含多分类）
zero-shot-classification	零样本分类
token-classification	命名实体识别 NER
ner	NER（token-classification 的别名）
question-answering	机器阅读理解 QA
table-question-answering	基于表格（如 TabFact）的 QA
fill-mask	Masked LM 填空任务
text-generation	文本生成（GPT 系）
text2text-generation	Seq2Seq 文本生成（T5、BART）
summarization	文本摘要
translation	机器翻译
translation_xx_to_yy	指定语种翻译，如 zh-en
conversational	对话机器人（DialoGPT）
feature-extraction	将文本转为 embedding 向量
sentence-similarity（新）	句子相似度
document-question-answering	文档型 QA（OCR+QA）
automatic-speech-recognition	语音识别 ASR（也算 NLP）
text-to-speech	TTS 文本转语音
zero-shot-audio-classification	零样本音频分类
🟩 CV（计算机视觉）任务
pipeline 名称	功能
image-classification	图像分类
object-detection	目标检测
image-segmentation	分割（包含 panoptic / semantic / instance）
semantic-segmentation	语义分割
instance-segmentation	实例分割
panoptic-segmentation	全景分割
image-to-text	图像→文本描述（BLIP 等）
image-feature-extraction	图像 embedding
image-retrieval	图文检索（如 CLIP）
depth-estimation	深度估计
zero-shot-image-classification	零样本图像分类
🟧 Audio（音频）
pipeline 名称	功能
automatic-speech-recognition	语音识别（如 Wav2Vec2）
audio-classification	音频分类
zero-shot-audio-classification	零样本音频分类
text-to-speech	文本转语音
speech-segmentation	语音分段
🟫 Multimodal（多模态）
pipeline 名称	功能
document-question-answering	文档理解（OCR+QA）
visual-question-answering	视觉问答（图像+文本）
image-to-text	图像描述
image-text-to-text	图像+文本 → 文本
video-classification	视频分类
zero-shot-image-classification	CLIP
zero-shot-audio-classification	音频零样本
🟪 特殊任务
pipeline 名称	功能
speech-to-speech	声音→声音（如 Seamless）
video-classification	视频分类
depth-estimation	深度图生成