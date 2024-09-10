import cv2
import time

# 카메라 초기화
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# 해상도 및 프레임 레이트 설정
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_interval = 1 / 15  # 목표 프레임 레이트: 15 FPS

while True:
    start_time = time.time()

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        # 프레임 처리 및 표시
        cv2.imshow('Camera 1', frame1)
        cv2.imshow('Camera 2', frame2)

    # 동기화 문제 해결을 위한 대기 시간
    elapsed_time = time.time() - start_time
    sleep_time = max(0, frame_interval - elapsed_time)
    time.sleep(sleep_time)  # 목표 FPS에 맞춰 대기

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
