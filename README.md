<p align="center">  
Install
Clone repo and install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.8.
</p>   

git clone https://github.com//xiaoli7766111/DRF  # clone
pip install -r requirements.txt  # install

BN_reg 为修剪策略一
BN_reg_st2 为修剪策略二
由于我们为了更贴合设计的正则化项，从头开始进行训练
Inference with main.py 得到预训练文件
Inference with finetune.py 使用组正则化项进行微调

使用 prune_vgg.py 对vgg模型进行压缩，使模型的计算量与存储量减少。
使用 prune_resnet.py 对resnet模型进行压缩，去除冗余的结构。

最后可选择性使用 retrain.py 恢复压缩对模型的影响 
