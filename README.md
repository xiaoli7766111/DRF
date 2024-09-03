# Install

<p align="center">  
  Clone repo and install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.8.
</p>   


```bash
git clone https://github.com/xiaoli7766111/DRF  # clone

pip install -r requirements.txt  # install
```

##  Train

**由于我们为了更贴合设计的组正则化项，从头开始进行训练**

BN_reg        为修剪策略一 <br>
BN_reg_st2    为修剪策略二 <br>
<br>

```bash
main.py
```
Inference with main 进行模型训练，得到预训练文件

##  Finetune
Inference with finetune 使用组正则化项进行微调
```bash
finetune.py
```

##  Pruning
prune_vgg.py 对Vgg模型进行压缩，使模型的计算量与存储量减少 <br>
prune_resnet.py 对ResNet模型进行压缩，去除冗余的结构
```bash
prune_vgg.py

prune_resnet.py
```

##  Retrain
最后可选择性使用 retrain 恢复压缩对模型的影响 
```bash
retrain.py
```
