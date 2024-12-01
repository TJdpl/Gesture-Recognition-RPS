# Gesture-Recognition-RPS
石头-剪刀-布手势实时识别（Pytorch and GUI）

## 1. 项目描述
本项目使用Pytorch框架搭建了一个简单的CNN模型，训练了三个手势的数据集，分别是石头、剪刀、布（以及人脸），用于训练模型。然后，使用PyQt5框架搭建了一个GUI界面，实时检测画面中的手势，并将结果反馈在一个文本框中。（作者的人工智能基础课程的大作业）  
![alt](.\demo\demo_img.png)
## 2. 项目简单介绍
1. 项目最初灵感来源于一个手势识别项目：
   https://github.com/AwakenPurity/Gesture-Recognition

2. 考虑到除了石头-剪刀-布三种手势外，在不摆手势的时候，人脸便会出现在画面中。因此，本实验中添加了人脸的数据集，用于在不摆手势时，将检测结果检测为“人”。但是目前人脸会识别成石头（未解决的问题）。

3. 为了实时检测，本项目设计了一个基于Opencv和PyQt5的GUI界面，支持实时画面来检测手势，实时画面直接将结果反馈在一个文本框中，并有一个退出按钮；按钮和文本里在画面右边，并将画面改成非镜像。

4. 在GUI界面右侧有游戏模式的按钮，点击后会倒计时3秒，电脑会随机生成一个手势与玩家进行石头剪刀布游戏，并告知结果。

## 3. 数据集
1. [瑞士Dalle Molle研究所数据](https://github.com/alessandro-giusti/rock-paper-scissors)。该项目包含在各种场合收集到的剪刀石头布图片，分为训练集（D1-D9）、测试集（test）

2. [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)。CelebA是一个大规模的名人人脸数据集，包含了大约20万张名人的面部图像。数据集中的人脸图像具有不同的姿势、表情和背景。

3. [飞桨al studio平台](https://aistudio.baidu.com/datasetdetail/87430).飞桨al studio平台上提供了丰富的手势数据集，包括手势数据集、人脸数据集、动作数据集等。其中手势数据集来自谷歌大脑数据集，包含了石头、剪刀、布三种手势的图片数据集。

4. 数据类别： 'paper'表示'布',  'people'表示'人脸',  'rock'表示'石头',  'scissor'表示'剪刀'。
```
{'paper': 0, 
 'people': 1, 
 'rock': 2, 
 'scissor': 3}
 ```
 
## 4. 项目执行
右击```main.py```文件，运行即可 或 在终端执行命令```python main.py```。
### datasets
文件夹里有四个数据集，分别是rock、paper、scissors、people。
### models
文件夹里有搭建的cnn模型，会被main.py自动调用。（还有一个老的残差网络模型，但效果不好）

### operation：
- ```train.py/.ipynb```：用于训练模型
- ```imageRename```: 用于重命名图片文件名
- ```img_size```: 用于读取图片大小
  
## 5. 需待改进
1. 准确率还有待提升
2. 受背景影响有点严重，手尽量占据大部分位置，背景最好不要有人头，会被稳定识别出石头
