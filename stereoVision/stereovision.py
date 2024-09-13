import cv2

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Open both cameras
cap_right = cv2.VideoCapture(1, cv2.CAP_DSHOW)                    
cap_left = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Create SGBM object
stereo = cv2.StereoSGBM_create(
    numDisparities=64,
    blockSize=23,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32     
)

while(cap_right.isOpened() and cap_left.isOpened()):
    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

    frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    gray1 = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.equalizeHist(gray1) #카메라 노출도 차이 히스토그램 평활화를 통해 보완
    gray2 = cv2.equalizeHist(gray2)

    blended = cv2.addWeighted(gray1,0.5,gray2,0.5,0)

    disparity = stereo.compute(gray1,gray2)

    disp8 = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_filtered = cv2.medianBlur(disp8, ksize=5)

    disp_colored = cv2.applyColorMap(disp_filtered, cv2.COLORMAP_JET)

    cv2.imshow("frame right", frame_right)
    cv2.imshow("frame left", frame_left)
    cv2.imshow('disparity', disp_colored)
    cv2.imshow('blended', blended) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()