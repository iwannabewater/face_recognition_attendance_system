import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime, date
import joblib

# 常量定义
MIN_TIME_BETWEEN_ATTENDANCE = 60  # 同一人出现的最小时间间隔

# 初始化目录并加载模型
if not os.path.isdir('attendance'):
    os.makedirs('attendance')  # 如果不存在attendance目录，则创建

if not os.path.isdir('static'):
    os.makedirs('static')  # 如果不存在static目录，则创建

if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')  # 如果不存在static/faces目录，则创建

model = joblib.load('static/face_recognition_model.pkl')  # 加载面部识别模型
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 加载OpenCV的人脸检测器

# 提取图像中的人脸的函数
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))  # 检测人脸
    return face_points

# 添加考勤记录的函数
def add_attendance(name, datetoday):
    username, userid = name.split('_')  # 分割获取用户名和用户ID
    current_time = datetime.now()  # 获取当前时间
    current_time_str = current_time.strftime("%H:%M:%S")  # 格式化当前时间

    attendance_file = f'attendance/Attendance-{datetoday}.csv'  # 考勤文件名
    df = pd.read_csv(attendance_file)  # 读取考勤文件
    last_attendance = df[df['Roll'] == int(userid)].tail(1)  # 获取最近的考勤记录

    # 检查是否符合考勤时间间隔
    if last_attendance.empty or (current_time - datetime.strptime(last_attendance['Time'].values[0], "%H:%M:%S")).seconds > MIN_TIME_BETWEEN_ATTENDANCE:
        with open(attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time_str}')  # 写入考勤记录

# 主函数，用于面部识别和考勤记录
def main():
    datetoday = date.today().strftime("%m_%d_%y")  # 获取当前日期
    attendance_file = f'attendance/Attendance-{datetoday}.csv'  # 定义当天的考勤文件

    # 如果当天的考勤文件不存在，则创建
    if not os.path.isfile(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Roll,Time')

    cap = cv2.VideoCapture(0)  # 打开摄像头

    while True:
        ret, frame = cap.read()  # 读取摄像头帧
        if not ret:
            break

        faces = extract_faces(frame)  # 提取人脸
        for (x, y, w, h) in faces:
            # 提取人脸并预测身份
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = model.predict(face.reshape(1, -1))[0]

            # 添加考勤记录
            add_attendance(identified_person, datetoday)

            # 绘制矩形框和文本
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, identified_person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 显示帧
        cv2.imshow('Attendance System', frame)

        # 按'Esc'键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

if __name__ == '__main__':
    main()
