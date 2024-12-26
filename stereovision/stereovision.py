# Basic module
import cv2
import numpy as np
import time
# Use to check distance
import matplotlib
import matplotlib.pyplot as plt

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Open both cameras
cap_left = cv2.VideoCapture(1)  # 1번 카메라, LEFT // 카메라에 붙인 number
cap_right = cv2.VideoCapture(0)   # 0번 카메라, RIGHT
# In Linux --> cv2.CAP_V4L2

# Filtering (Morphological filter)
kernel = np.ones((3, 3), np.uint8)

# Prepare parameter (use trackbar)
# nDispFactor = 12  # adjust this
# window_size = 7
# min_disp = 16  # 가까운 물체를 인식 안하는 방향

# Sharpening kernel (Laplacian)
sharpening_kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])

sharpening_kernel_2 = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

# 동기화를 위한 프레임 레이트 설정
frame_interval = 1/15 # 목표 프레임 레이트 : 15FPS

# Make trackbar
def onChange(pos):
    pass

cv2.namedWindow('disparity_trackbar',cv2.WINDOW_NORMAL)
cv2.createTrackbar('min_disparity','disparity_trackbar',0,31, onChange)
cv2.createTrackbar('num_disparity','disparity_trackbar',0,31, onChange)
cv2.createTrackbar('blocksize','disparity_trackbar',0,31, onChange)
cv2.createTrackbar('disp','disparity_trackbar',0,15, onChange)
cv2.createTrackbar('specklesize','disparity_trackbar',63,511, onChange)
cv2.createTrackbar('uniqueRatio','disparity_trackbar',0,63, onChange)
cv2.createTrackbar('speckleR','disparity_trackbar',0,63, onChange)

cv2.setTrackbarPos('disp','disparity_trackbar',0)
cv2.setTrackbarPos('min_disparity','disparity_trackbar',16)
cv2.setTrackbarPos('num_disparity','disparity_trackbar',12)
cv2.setTrackbarPos('blocksize','disparity_trackbar',6)
cv2.setTrackbarPos('specklesize','disparity_trackbar',200)
cv2.setTrackbarPos('uniqueRatio','disparity_trackbar',8)
cv2.setTrackbarPos('speckleR','disparity_trackbar',16)

# Starting the stereo vision
while(cap_right.isOpened() and cap_left.isOpened()):
    start_time = time.time()

    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

    frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    grayR = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)

    # 조명 환경이 불균형일 때
    grayR = cv2.equalizeHist(grayR) # 카메라 노출도 차이 히스토그램 평활화를 통해 보완
    grayL = cv2.equalizeHist(grayL)

    # Apply Gaussian filter
    grayR_smoothed = cv2.GaussianBlur(grayR, (9,9), 10)
    grayL_smoothed = cv2.GaussianBlur(grayL, (9,9), 10)

    # addweight image
    grayR = cv2.addWeighted(grayR, 1.5, grayR_smoothed, -0.5, 0)
    grayL = cv2.addWeighted(grayL, 1.5, grayL_smoothed, -0.5, 0)

    # Apply medianBlur filter
    # grayR = cv2.medianBlur(grayR, ksize=5)
    # grayL = cv2.medianBlur(grayL, ksize=5)

    # Apply sharpening filter to both images
    # grayR = cv2.filter2D(grayR, -1, sharpening_kernel)
    # grayL = cv2.filter2D(grayL, -1, sharpening_kernel)

    # 여러 필터를 적용 결과 이미지 값 받아오는 건 위 과정이 제일 최적인 듯

    # 합쳐진 형상이 다음과 같음
    blended = cv2.addWeighted(grayL,0.5,grayR,0.5,0)

    # get trackbar parameter
    min_disp = cv2.getTrackbarPos('min_disparity','disparity_trackbar') # default : 16
    nDispFactor = cv2.getTrackbarPos('num_disparity','disparity_trackbar') # default : 12
    window_size = cv2.getTrackbarPos('blocksize','disparity_trackbar') # default : 6
    disp_max = cv2.getTrackbarPos('disp','disparity_trackbar') # default : 0
    specklesize = cv2.getTrackbarPos('specklesize','disparity_trackbar') # default : 200  // 조정이 필요할 듯? 높을수록 노이즈가 줄어드는 경향성이 존재
    uniqueRatio = cv2.getTrackbarPos('uniqueRatio','disparity_trackbar') # default : 8
    speckleR = cv2.getTrackbarPos('speckleR','disparity_trackbar') # default : 16

    # Create SGBM object
    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = 16 * nDispFactor - min_disp,  # 두 이미지 간의 최대 시차값을 설정, 16의 배수 --> 값이 클수록 멀리 있는 물체 감지
        blockSize = window_size,  # 매칭 블록 크기 설정
        P1 = 8 * 3 * window_size ** 2,  # setting smoothness
        P2 = 32 * 3 * window_size ** 2,
        disp12MaxDiff = disp_max,  # 두좌우 disparity 맵의 차이 허용 범위
        uniquenessRatio = uniqueRatio,  # 두 매칭된 점들 간의 품질을 비교해 최상의 매칭만 허용하는 정도
        speckleWindowSize = specklesize,  # 작은 노이즈 영역을 제거하기 위함
        speckleRange = speckleR  # 인접한 픽셀 사이의 disparity 값이 허용할 수 있는 최대 차이
    )

    # stereo compute 시 왼쪽 화면 기반
    disparity = stereo.compute(grayL,grayR)
    disp8 = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Using the medianBlur
    disp_filtered = cv2.medianBlur(disp8, ksize=3)

    # Morphological Filter 적용
    closing = cv2.morphologyEx(disp_filtered, cv2.MORPH_CLOSE, kernel)
    # 확실히 노이즈가 줄어드는건 맞는데 필터 적용 전이 더 자연스러운 느낌이긴 함.

    # colorMap 적용
    disp_colored = cv2.applyColorMap(disp_filtered, cv2.COLORMAP_OCEAN)
    closing_colored = cv2.applyColorMap(closing,cv2.COLORMAP_OCEAN)

    # 적응형 이진화
    # adaptive_thresh = cv2.adaptiveThreshold(closing, 255,
    #                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                         cv2.THRESH_BINARY,
    #                                         11, 2)

    cv2.imshow("calibration_cap_right", grayR)
    cv2.imshow("calibration_cap_left", grayL)
    cv2.imshow('disparity_map', closing)
    cv2.imshow('disparity_map_closing', closing_colored)

    # 동기화 문제 해결을 위한 대기 시간
    elapsed_time = time.time()-start_time
    sleep_time = max(0, frame_interval - elapsed_time)
    time.sleep(sleep_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()

### 사용 필터 점검 ###
# wls filter : 연산량이 너무 많아 프레임 드랍이 심해 사용 X
# sharpening filter : 오히려 이미지가 부자연스러워지는 느낌