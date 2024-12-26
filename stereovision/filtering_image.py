import cv2
import os
import glob

# 입력 폴더와 출력 폴더 경로 설정
input_folder = 'real_sequence/image_r'
output_folder = 'real_sequence/image_filtered_r'

# 출력 폴더가 없다면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 특정 폴더에서 모든 이미지 파일 경로 불러오기
image_paths = glob.glob(os.path.join(input_folder, '*.png'))  # .jpg 이미지를 대상으로


# 이미지 필터링 함수 (예: 가우시안 블러)
def filter_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image_smoothed = cv2.GaussianBlur(image, (9,9), 10)
    image = cv2.addWeighted(image, 1.5, image_smoothed, -0.5, 0)
    return image


# 이미지 처리 및 저장
for image_path in image_paths:
    # 이미지 불러오기
    img = cv2.imread(image_path)
    # 필터링 진행
    filtered_img = filter_image(img)

    # 출력 파일 경로 설정 (이미지 이름 유지)
    output_path = os.path.join(output_folder, os.path.basename(image_path))

    # 결과 이미지 저장
    cv2.imwrite(output_path, filtered_img)

    print(f'Processed and saved: {output_path}')
