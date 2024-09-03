# Install

<p align="center">  
 Clone repo and install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.8.
</p>   


```bash
git clone https://github.com/xiaoli7766111/DRF  # clone

pip install -r requirements.txt  # install
```

##  Train

**BN_reg is the code for pruning strategy I of ResNets.**<br> **BN_reg_st2 is the code for pruning strategy II of ResNets.**

<br>

```bash
main.py
```
**Training with main.py**

##  Finetune
**finetuning with finetune.py**<br>
```bash
finetune.py
```

##  Pruning
 **Using prune_vgg.py to prune VGGNet** <br>
**Using prune_resnet.py to prune ResNet** <br>
```bash
prune_vgg.py

prune_resnet.py
```

##  Retrain
**Finally, the impact of compression on the model can be selectively restored using retrain.py** <br>
```bash
retrain.py
```
