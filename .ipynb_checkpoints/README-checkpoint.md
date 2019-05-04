# 版本
Github上的代码会不定期进行更新，将会逐渐支持以下软件版本  
* Python-3.6.8  
* PyTorch-1.0.1

猫狗数据集Kaggle下载地址：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data  
猫狗数据集百度网盘下载地址：https://pan.baidu.com/s/1yKpDPRfFkYca6E1nZ-g1hg 提取码：qh1z 


# 本书纠错

感谢**xhy3054、Zehui-Lin**发现书中的错误。

## 第1章  

P2   
原文：  
...，人工智能历史上的第1股浪潮就这样顺理成章**地**形成了，...  
纠正：  
...，人工智能历史上的第1股浪潮就这样顺理成章的形成了，... 


## 第2章  

P17  
原文：  
如果我们想要获得索引值是**12**和**33**的值，...  
纠正：  
如果我们想要获得索引值是21和32的值，...  


## 第3章  

P40  
原文：  
...，虽然梯度下降已被广泛应用，但是其自身**纯**在许多不足，...  
纠正：  
...，虽然梯度下降已被广泛应用，但是其自身存在许多不足，...  

## 第5章  

P77  
原文：  
（3）conda remove -n test **-all**：在命令行窗口中输入“conda remove -n test **-all**”并回车，... ；“**-all**”表示删除指定环境下所有已经安装的包。  
纠正：  
（3）conda remove -n test --all：在命令行窗口中输入“conda remove -n test --all”并回车，...  ；“--all”表示删除指定环境下所有已经安装的包。 

P121  
原文：  
（1）使用NumPy中的**onse**可以创建维度指定且元素全为1的数组。...，在以上代码中使用**onse**生成了一个元素全为1且维度为（2,3）的数组，传递给**onse**的参数是一个列表，...  
纠正：  
（1）使用NumPy中的ones可以创建维度指定且元素全为1的数组。...，在以上代码中使用ones生成了一个元素全为1且维度为（2,3）的数组，传递给ones的参数是一个列表，...   


P129  
原文：  
（3）randn：生成一个满足平均值为0且方差为1的正**太**分布随机样本数。...  
纠正：  
（3）randn：生成一个满足平均值为0且方差为1的正态分布随机样本数。...   

P130  
原文：  
（7）normal：生成一个指定维度且满足高斯正**太**分布的随机样本数。...  
纠正：  
（7）normal：生成一个指定维度且满足高斯正态分布的随机样本数。...   

P140  
原文：  
（5）“loc=4”：强制图例使用图中右**上**角的位置。...  
纠正：  
（5）“loc=4”：强制图例使用图中右下角的位置。...   


## 第6章

P145  
原文：  
...，随机生成的浮点数的取值满足均值为0、方差为1的正**太**分布。...  
纠正：  
...，随机生成的浮点数的取值满足均值为0、方差为1的正态分布。...   

P151  
原文：  
...，即前一个矩阵的***行***数必须和后一个矩阵的***列***数相等，...     
纠正：  
...，即前一个矩阵的列数必须和后一个矩阵的行数相等，...     


P171  
原代码：  
```python
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
```
纠正代码：  
```python
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5],std=[0.5])])
```

P173  
原代码：  
```python
images, labels = next(iter(data_loader_train))

img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1,2,0)

std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std+mean
print([labels[i] for i in range(64)])
plt.imshow(img)
```
纠正代码：  
```python
images, labels = next(iter(data_loader_train))

img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1,2,0)

std = [0.5]
mean = [0.5]
img = img*std+mean
print([labels[i] for i in range(64)])
plt.imshow(img)
```

P177  
原代码：  
```python
loss.backward()
optimizer.step()
running_loss += loss.data[0]
running_correct += torch.sum(pred == y_train.data)
```
纠正代码：
```python
loss.backward()
optimizer.step()
running_loss += loss.data
running_correct += torch.sum(pred == y_train.data)
```


## 第7章

P216-217   
原代码：  
```python
path = "dog_vs_cat"
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
```
纠正代码：
代码冗余进行删除。

## 第9章

P234    
原文：  
...，如果为偶数，**则极有可能会出现结果无法判断的情况**。  
纠正： 
...，如果为偶数，还需要多一步结果的随机抽选。 


P236    
原文：  
...，比如在Kaggle比赛中就经常会用到各种各样的多模型融合实例。  
纠正： 
...，比如在Kaggle比赛中就经常会用到各种各样的多模型融合~~实例~~。 

## 第10章

P256   
原代码：  
```python
print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss/len(dataset_train),100*running_correct/len(dataset_train),100*testing_correct/len(dataset_test)))
```
纠正代码：
代码前多加4个空格。

## 第11章

P264   
原代码：  
```python
img1 = torchvision.utils.make_grid(X_test)
img1 = img1.numpy().transpose(1,2,0)
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img1 = img1*std+mean
```
纠正代码：
```python
img1 = torchvision.utils.make_grid(X_test)
img1 = img1.numpy().transpose(1,2,0)
std = [0.5]
mean = [0.5]
img1 = img1*std+mean
```

P265、P271   
原代码：  
```python
transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])


std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
```
纠正代码：
```python
transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5], std=[0.5])])


std = [0.5]
mean = [0.5]
```

P273  
原代码：  
```python
print("Loss is:{:.4f}".format(running_loss/len(dataset_train)))
```
纠正代码：
代码前多加4个空格。


---

# 《深度学习之PyTorch实战计算机视觉》全书代码

## 作者其他链接
作者知乎主页：https://www.zhihu.com/people/JaimeTang/activities   

## 问题 
读者如果发现书中有问题，希望提issues给本人，作者会及时解答，谢谢! 

## 封面
![简介](image/10.jpg)  
