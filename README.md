# 版本
Github上的代码会不定期进行更新，将会逐渐支持以下软件版本  
* Python-3.6.8  
* PyTorch-1.0.1

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


## 第6章

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

---

# 《深度学习之PyTorch实战计算机视觉》全书代码

## 使用的数据集
猫狗数据集下载地址：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

## 作者其他链接
作者知乎主页：https://www.zhihu.com/people/JaimeTang/activities   

## 问题 
读者如果发现书中有问题，希望提issues给本人，作者会及时解答，谢谢! 

## 封面
![简介](image/10.jpg)  
