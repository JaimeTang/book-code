# 本书纠错

感谢**xhy3054、Zehui-Lin**发现书中的错误。

## 第3章  

P40  
原文：  
...，虽然梯度下降已被广泛应用，但是其自身**纯**在许多不足，...  
纠正：  
...，虽然梯度下降已被广泛应用，但是其自身存在许多不足，...  

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

## 版本
以下版本可以成功运行书中全部代码  
* Python-3.5  
* PyTorch-0.4.1

## 使用的数据集
猫狗数据集下载地址：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

## 作者其他链接
作者知乎主页：https://www.zhihu.com/people/JaimeTang/activities   

## 问题 
读者如果发现书中有问题，希望提issues给本人，作者会及时解答，谢谢! 

## 封面
![简介](image/10.jpg)  
