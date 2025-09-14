# Visual Odometry (with IMU)

진행 과정은 notion에 기록 
https://www.notion.so/visual-slam-108ff3f9b083809eb7aafcf0c0d6149d

stereo vision 구현을 통한 visual slam 

usb 웹캠 2개 사용 예정 


trigger 기능 X --> 카메라 '동기화'가 매우 중요한 point

어떻게 카메라를 동기화해야할까? 

웹캠은 ABKO사의 APC850 사용 예정 

step 1) 카메라 캘리브레이션 진행 
calibration을 바탕으로 stereoMap.xml(왜곡계수)와 calib_test.txt(카메라 프로젝션 행렬)을 도출함

step 2) stereo vision 구현
SGBM_create()의 적절한 parameter를 찾기 위함

step 3) sensor fusion
sensor fusion을 위한 IMU 데이터를 불러옴 

step 4) visual inertial odometry
위 내용 바탕으로 VIO 진행 


참고 github
https://github.com/LearnTechWithUs/Stereo-Vision
