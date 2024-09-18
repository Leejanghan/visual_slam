import cv2

#스테레오비전 캘리브레이션을 위한 이미지 수집용 코드
cap1 = cv2.VideoCapture(0)  # 1번 카메라
cap2 = cv2.VideoCapture(1)  # 0번 카메라
# In Linux --> cv2.CAP_V4L2 라는 default값을 추가하는 것이 좋음

num = 0

# checking camera_1
if not cap1.isOpened():
    print('unable to read camera_1 feed')

# checking camera_2
if not cap2.isOpened():
    print('unable to read camera_2 feed')

# recieve image data
while cap1.isOpened() and cap2.isOpened():
    success1, img1 = cap1.read()
    success2, img2 = cap2.read()

    k = cv2.waitKey(5)

    # Mean Esc
    if k == 27:
        break
    # Wait for 's' key to save and exit
    elif k == ord('s'):
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img1)
        cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', img2)
        print("images saved!")
        num += 1

    cv2.imshow('Img 1',img1)
    cv2.imshow('Img 2',img2)

cap1.release()
cap2.release()

cv2.destroyAllWindows()

### 시간 동기화가 잘 이루어지지 않는다면, 목표 프레임 레이트를 설정해야할 수 있음
