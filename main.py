import sys 
import torch 
import cv2 
import random  # 导入random库，用于生成随机数
from PyQt5.QtCore import Qt, QTimer 
from PyQt5.QtGui import QImage, QPixmap 
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox  # 导入QMessageBox用于显示消息框
from torchvision.transforms import transforms 

# 加载训练好的模型 
from models.cnn import ConvNet 
model = ConvNet() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load('best_model.pt')) 
model.eval() 

# 定义手势标签
gesture_label = ["布", "人", "石头", "剪刀"]

class GestureRecognitionApp(QMainWindow): 
    def __init__(self): 
        super().__init__() 
        self.setWindowTitle("Gesture Recognition RPS")  # 设置窗口标题
        self.setGeometry(100, 100, 800, 600)  # 设置窗口的初始位置和大小
        
        self.videoLabel = QLabel(self)  # 创建一个QLabel用于显示视频画面
        self.resultTextEdit = QTextEdit(self)  # 创建一个QTextEdit用于显示识别结果
        self.exitButton = QPushButton("退出", self)  # 创建一个QPushButton用于退出应用
        
        # 添加“进行游戏”按钮
        self.playButton = QPushButton("进行游戏", self)
        self.playButton.clicked.connect(self.startGame)
        
        layout = QHBoxLayout() 
        layout.addWidget(self.videoLabel) 
        
        rightLayout = QVBoxLayout() 
        rightLayout.addWidget(self.exitButton)
        rightLayout.addWidget(self.playButton)  # 将“进行游戏”按钮添加到右侧布局
        rightLayout.addWidget(self.resultTextEdit) 
        
        layout.addLayout(rightLayout) # 将垂直布局添加到水平布局中
        
        widget = QWidget(self)  # 创建一个QWidget作为中央部件的容器
        widget.setLayout(layout)  # 为容器设置布局
        self.setCentralWidget(widget)  # 将容器设置为窗口的中央部件
        
        self.capture = cv2.VideoCapture(0)  # 创建一个VideoCapture对象，用于捕获摄像头视频
        self.timer = QTimer(self)  # 创建一个QTimer定时器
        self.timer.timeout.connect(self.showFrame)  # 将定时器的timeout信号连接到showFrame槽函数
        self.timer.start(30)  # 启动定时器，每隔30毫秒触发一次timeout信号
        
        self.exitButton.clicked.connect(self.exit) # 将退出按钮的clicked信号连接到exit槽函数
        
        # 游戏相关的变量
        self.game_running = False
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.countdown)
        
    def showFrame(self): 
        # 显示视频画面
        ret, frame = self.capture.read() 
        if ret: 
            frame = cv2.flip(frame, 1) 
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            h, w, ch = rgb_image.shape 
                
        if not self.game_running:  # 如果游戏未运行，则正常显示摄像头画面
            # 调整图像大小为模型需要的输入大小
            transform = transforms.Compose([ 
                transforms.ToPILImage(),  # 将图像转换为PIL格式
                transforms.Resize((500, 500)),  # 将图像大小调整为500x500
                transforms.ToTensor(),  # 将图像转换为Tensor格式
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 对图像进行标准化处理
            ]) 
            image = transform(rgb_image)  # 对图像应用预处理
            image = image.unsqueeze(0)  # 增加一个维度，使图像的形状符合模型输入要求 
            
            # 使用模型进行手势识别
            with torch.no_grad(): 
                image = image.to(device)  # 将图像转移到GPU设备
                output = model(image) 
                _, predicted = torch.max(output, 1) 
            
            # 显示手势标签
            result = gesture_label[predicted.item()] 
            # 在实时画面上绘制识别结果
            cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 
            # 在文本框中显示识别结果
            self.resultTextEdit.setText(result) 
            # 显示处理后的图像
            q_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888) # 将RGB图像数据转换为QImage对象
            self.videoLabel.setPixmap(QPixmap.fromImage(q_image)) # 将QImage对象转换为QPixmap对象并显示在QLabel上
        else:
            # 游戏进行中，只显示摄像头画面，不做实时识别
            q_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888) # 将RGB图像数据转换为QImage对象
            self.videoLabel.setPixmap(QPixmap.fromImage(q_image)) # 将QImage对象转换为QPixmap对象并显示在QLabel上
            pass
    
    def exit(self): 
        self.capture.release() 
        self.close() 
        
    def startGame(self):
        # 开始游戏，重置游戏状态
        self.game_running = True
        self.countdown_value = 3  # 倒计时3秒
        self.countdown_timer.start(1000)  # 每秒触发一次倒计时
        self.resultTextEdit.setText("倒计时：3")  # 显示倒计时
        
    def countdown(self):
        # 倒计时函数
        if self.countdown_value > 0:
            self.countdown_value -= 1
            self.resultTextEdit.setText(f"倒计时：{self.countdown_value}")
        else:
            self.countdown_timer.stop()
            self.game_running = False  # 停止倒计时，设置游戏为未运行状态
            # 随机生成石头、剪刀、布
            computer_choice = random.choice(gesture_label)
            while computer_choice == "人":
                computer_choice = random.choice(gesture_label)  # 若电脑随机选择“人”，则重新随机选择
            
            # 这里需要暂停一下，以便捕捉当前摄像头画面进行识别（模拟）
            # 实际使用时，应该捕获并识别当前摄像头画面
            # 假设已经识别出用户的手势为user_choice（需要从showFrame函数中获取）
            # 但由于showFrame在另一个线程中运行，需要一些同步机制（例如使用信号槽）
            ret, frame = self.capture.read()  # 从摄像头读取一帧图像
            if ret:
                frame = cv2.flip(frame, 1)  # 将图像水平翻转，以非镜像方式显示
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将图像从BGR格式转换为RGB格式

                # 调整图像大小为模型需要的输入大小
                transform = transforms.Compose([
                    transforms.ToPILImage(),  # 将图像转换为PIL格式
                    transforms.Resize((500, 500)),  # 将图像大小调整为500x500
                    transforms.ToTensor(),  # 将图像转换为Tensor格式
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 对图像进行标准化处理
                ])
                image = transform(rgb_image)  # 对图像应用预处理
                image = image.unsqueeze(0)  # 增加一个维度，使图像的形状符合模型输入要求
    
                # 使用模型进行手势识别
                with torch.no_grad():  # 在不需要计算梯度的情况下进行推理
                    image = image.to(device)  # 将图像转移到GPU设备
                    output = model(image)  # 将预处理后的图像输入模型进行推理
                    _, predicted = torch.max(output, 1)  # 获取预测结果中得分最高的类别
    
                user_choice = gesture_label[predicted.item()]  # 获取预测的手势标签
                
                
            # 判断胜负
            if user_choice == computer_choice:
                result = "平局"
            elif (user_choice == "布" and computer_choice == "石头") or \
                 (user_choice == "石头" and computer_choice == "剪刀") or \
                 (user_choice == "剪刀" and computer_choice == "布"):
                result = "胜"
            elif (user_choice == "人"):
                result = "未出手"
            else:
                result = "负"
            
            # 显示结果
            QMessageBox.information(self, "游戏结果", f"你选择了：{user_choice}\n电脑选择了：{computer_choice}\n结果：{result}")
            
            # 提供“再来一局”和“退出游戏”选项
            reply = QMessageBox.question(self, '游戏结束', 
                                         "你想再来一局还是退出游戏？\n点击‘是’再来一局，点击‘否’退出游戏。", 
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                # 再来一局，重置游戏状态
                self.startGame()
            else:
                # 退出游戏，停止摄像头
                self.game_running = False

if __name__ == "__main__": 
    app = QApplication(sys.argv) 
    window = GestureRecognitionApp() 
    window.show() 
    sys.exit(app.exec_())