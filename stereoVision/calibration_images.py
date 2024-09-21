import cv2
import os

#스테레오비전 캘리브레이션을 위한 이미지 수집용 코드
cap_left = cv2.VideoCapture(0)  # 1번 카메라, LEFT
cap_right = cv2.VideoCapture(1)  # 0번 카메라, RIGHT
# In Linux --> cv2.CAP_V4L2

num = 0

# 디렉토리 생성 함수
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

create_dir_if_not_exists('images/stereoLeft')
create_dir_if_not_exists('images/stereoRight')

# checking camera_1
if not cap_left.isOpened():
    print('unable to read camera_left feed')

# checking camera_2
if not cap_right.isOpened():
    print('unable to read camera_right feed')

# Receive image data
while cap_left.isOpened() and cap_right.isOpened():
    success1, img_left = cap_left.read()
    success2, img_right = cap_right.read()

    k = cv2.waitKey(5)

    # Mean Esc
    if k == 27:
        break
    # Wait for 's' key to save and exit
    elif k == ord('s'):
        left_image = 'images/stereoLeft/imageL' + str(num) + '.png'
        right_image = 'images/stereoRight/imageR' + str(num) + '.png'

        # checking process
        if cv2.imwrite(left_image, img_left):
            print(f"Left image saved as {left_image}")
        else:
            print(f"Failed to save left image {left_image}")

        if cv2.imwrite(right_image, img_right):
            print(f"Right image saved as {right_image}")
        else:
            print(f"Failed to save right image {right_image}")
        num += 1

    cv2.imshow('Img left',img_left)
    cv2.imshow('Img right',img_right)

cap_left.release()
cap_right.release()

cv2.destroyAllWindows()

### 시간 동기화가 잘 이루어지지 않는다면, 목표 프레임 레이트를 설정해야할 수 있음 ###
