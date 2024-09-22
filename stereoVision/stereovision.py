# Basic module
import cv2
import numpy as np
# Use to check distance
import matplotlib
import matplotlib.pyplot as plt

# To check depth (Need to change disp type)
""" def coords_mouse_disp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print x,y,disp[y,x],filteredImg[y,x]
        average = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += disp[y + u, x + v]
        average = average / 9
        Distance = -593.97 * average ** (3) + 1506.8 * average ** (2) - 1373.1 * average + 522.06
        Distance = np.around(Distance * 0.01, decimals=2)
        print('Distance: ' + str(Distance) + ' m')
"""

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Open both cameras
cap_left = cv2.VideoCapture(0)  # 1번 카메라, LEFT // 카메라에 붙인 number
cap_right = cv2.VideoCapture(1)   # 0번 카메라, RIGHT
# In Linux --> cv2.CAP_V4L2

# Filtering
kernel = np.ones((3, 3), np.uint8)

# Prepare parameter
nDispFactor = 8  # adjust this
window_size = 12
min_disp = 5  # 가까운 물체를 인식 안하는 방향 --> 5가 최대 적정 값일듯?

# Create SGBM object
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = 16 * nDispFactor,  # 두 이미지 간의 최대 시차값을 설정, 16의 배수 --> 값이 클수록 멀리 있는 물체 감지
    blockSize = window_size, # 매칭 블록 크기 설정
    P1=8 * 3 * window_size ** 2, # setting smoothness
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff = 3, # 두 시차 맵 간의 최대 허용 차이 설정
    uniquenessRatio = 10, # 값이 클수록 매칭 난이도 올라감
    speckleWindowSize = 100, # 작은 노이즈 영역을 제거하기 위함
    speckleRange = 32
)

# Used for the filtered image
# stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

# WLS FILTER Parameters
# lmbda = 80000
# sigma = 1.8
# visual_multiplier = 1.0
# 만약, 프레임이 너무 떨어진다면 위 파라미터값을 조정해야함

# wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
# wls_filter.setLambda(lmbda)
# wls_filter.setSigmaColor(sigma)
# wls filter <-- 연산량이 너무 많아 프레임이 떨어짐

# Starting the stereo vision
while(cap_right.isOpened() and cap_left.isOpened()):
    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

    frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    grayR = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)

    # 조명 환경이 불균형일 때
    # grayR = cv2.equalizeHist(grayR) # 카메라 노출도 차이 히스토그램 평활화를 통해 보완
    # grayL = cv2.equalizeHist(grayL)

    # 합쳐진 형상이 다음과 같음
    blended = cv2.addWeighted(grayL,0.5,grayR,0.5,0)

    # stereo compute 시 왼쪽 화면 기반
    disparity = stereo.compute(grayL,grayR)
    disparity_Left = disparity

    # 오른쪽 화면 기반
    # disparity_Right = stereo.compute(grayR,grayL)

    # disparity_Left = np.int16(disparity_Left)
    # disparity_Right = np.int16(disparity_Right)

    # Using the WLS filter
    # filteredImg = wls_filter.filter(disparity_Left, grayL, None, disparity_Right)
    # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    # filteredImg = np.uint8(filteredImg)

    disp8 = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_filtered = cv2.medianBlur(disp8, ksize=5)

    # Morphological Filter 적용
    closing = cv2.morphologyEx(disp_filtered, cv2.MORPH_CLOSE, kernel)

    disp_colored = cv2.applyColorMap(closing, cv2.COLORMAP_OCEAN)

    cv2.imshow("calibration_cap_right", frame_right)
    cv2.imshow("calibration_cap_left", frame_left)
    cv2.imshow('disparity_map', disp_colored)
    cv2.imshow('blended', blended)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
