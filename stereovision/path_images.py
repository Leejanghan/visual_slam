import cv2
import os
import serial

# 스테레오비전 캘리브레이션을 위한 이미지 수집용 코드
cap_left = cv2.VideoCapture(1)  # 1번 카메라, LEFT
cap_right = cv2.VideoCapture(0)  # 0번 카메라, RIGHT
# In Linux --> cv2.CAP_V4L2q

num = 0

# 디렉토리 생성 함수
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

create_dir_if_not_exists('real_sequence/image_l')
create_dir_if_not_exists('real_sequence/image_r')

# 아두이노와 연결
def initialize_imu(port='/dev/ttyUSB0', baudrate=115200):
    ser = serial.Serial(port, baudrate, timeout=1)
    print(f"Connected to IMU on {port} at {baudrate} baud.")
    return ser

def read_yaw_from_imu(serial_connection):
    line = serial_connection.readline().decode('utf-8').strip()
    try:
        _, _, _, yaw = map(float, line.split())
        return yaw
    except ValueError:
        print("Invalid IMU data received.")
        return None

# IMU 초기화
imu_serial = initialize_imu(port='COM4')  # Windows에서는 COM 포트를 지정, Linux에서는 '/dev/ttyUSB0'

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
    yaw = read_yaw_from_imu(imu_serial)
    k = cv2.waitKey(5)

    # Mean Esc
    if k == 27:
        break

    # Wait for 's' key to save and exit
    elif k == ord('s'):
        left_image = 'real_sequence/image_l/' + str(num).zfill(6) + '.png'
        right_image = 'real_sequence/image_r/' + str(num).zfill(6) + '.png'

        # 이미지를 그레이스케일로 변환
        img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # checking process
        if cv2.imwrite(left_image, img_left_gray):
           print(f"Left image saved as {left_image}")
        else:
           print(f"Failed to save left image {left_image}")

        if cv2.imwrite(right_image, img_right_gray):
           print(f"Right image saved as {right_image}")
        else:
           print(f"Failed to save right image {right_image}")

        # IMU Yaw 값을 텍스트 파일에 저장
        if yaw is not None:
            with open('real_sequence/imu_data.txt', 'a') as imu_file:
                imu_file.write(f"{yaw}\n")
            print(f"IMU Yaw saved: {yaw}")
        else:
            print(f"IMU Yaw not saved")
        num += 1

    cv2.imshow('Img left',img_left)
    cv2.imshow('Img right',img_right)

cap_left.release()
cap_right.release()

cv2.destroyAllWindows()

# visual odometry를 위한 image 촬영