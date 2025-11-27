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