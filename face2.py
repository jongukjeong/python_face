import numpy as np
import cv2
import sys

# 입력 파일
image_file = "./man.jpg"

# 캐스케이드 파일의 경로 지정하기
cascade_file = "/usr/local/Cellar/opencv/4.1.1_2/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml"

# 이미지 읽어 들이기
image = cv2.imread(image_file)
print("---------->")
print(image)
print("<----------")

# 그레이스케일로 변환하기
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 인식 특징 파일 읽어 들이기
cascade = cv2.CascadeClassifier(cascade_file)
# 얼굴 인식 실행하기
face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=1, minSize=(150, 150))

if len(face_list) > 0:
    # 인식한 부분 표시하기
    print(face_list)
    color = (0, 0, 255)
    for face in face_list:
        x, y, w, h = face
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=8)
    # 파일로 출력하기
    cv2.imwrite("man-facedetect.jpg", image)
else:
    print("no face")
