# PyTorch
To Learn PyTorch

docker pull pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
cd C:\LocalGit\PyTorch (the project folder)
docker run -it --rm -v ${pwd}/:/workspace pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
pip install pandas==2.1.4 torchtext==0.17.2 torchdata==0.7.1 portalocker==2.0.0


http://neuralnetworksanddeeplearning.com/

git clone https://github.com/mnielsen/neural-networks-and-deep-learning.git

https://www.runoob.com/pytorch/pytorch-tutorial.html

to dig version conflict:
python -V
pip list | grep -E "torch|torchtext"
ldd /opt/conda/lib/python3.10/site-packages/torchtext/lib/libtorchtext.so


HuggingFace
docker pull huggingface/transformers-pytorch-cpu
cd C:\LocalGit\PyTorch (the project folder)
docker run -it --rm -v ${pwd}/:/workspace huggingface/transformers-pytorch-cpu