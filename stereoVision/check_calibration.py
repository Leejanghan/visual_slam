import cv2

# Load stereo calibration parameters
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FileStorage_READ)

if not cv_file.isOpened():
    print("Failed to open the FileStorage.")
    exit()

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

cv_file.release()

# Open both cameras
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

while cap_right.isOpened() and cap_left.isOpened():

    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

    if not success_right or not success_left:
        print("Failed to capture images from one or both cameras.")
        break
        
    # original images (only left)    
    cv2.imshow("original left", frame_left)

    # Undistort and rectify images
    frame_left_rec = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

    # Show the frames
    cv2.imshow("frame left", frame_left_rec)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy all windows before termination
cap_left.release()
cv2.destroyAllWindows()


## use only left camera to check calibration
