# visual_slam 

stereo vision 구현을 통한 visual slam 
usb 웹캠 2개 사용 예정 


trigger 기능 X --> 카메라 '동기화'가 매우 중요한 point
어떻게 카메라를 동기화해야할까? 

웹캠은 ABKO사의 APC850 사용 예정 

step ) 카메라 캘리브레이션 진행 
1. calibartion_images.py 실행
2. stereovisioncalibration.py 실행 
---> 진행할 때 이미지 저장을 잘해야 됨

step ) stereo vision 구현
1. camera calibration 먼저 진행
2. stereo vision.py 실행 
