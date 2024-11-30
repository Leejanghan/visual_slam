# 현재 코딩 중
import os
import numpy as np
import cv2
from scipy.optimize import least_squares
from lib.visualization import plotting
from lib.visualization.video import play_trip
from tqdm import tqdm

# Use two camera & None loop
class VisualOdometry():
    def __init__(self, data_dir):
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(data_dir + '/calib_test.txt')
        self.images_l = self._load_images(data_dir + '/image_l')
        self.images_r = self._load_images(data_dir + '/image_r')
        self.IMU = self._load_data(data_dir+'/imu_data.txt')

        # setting parameter
        window_size = 6
        min_disp = 16
        nDispFactor = 12
        disp_max = 0
        uniqueRatio = 8
        specklesize = 200
        speckleR = 16

        # SGBM 알고리즘을 통해 스트레오 매칭을 위한 disparity 설정을 수행
        self.disparity = cv2.StereoSGBM_create(
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
        self.disparities = [
            np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]

        # FAST 알고리즘을 사용하여 특징점을 검출함 <-- 성능만 생각하면 ORB 특징 추출이 더 좋을 듯?
        self.fastFeatures = cv2.FastFeatureDetector_create()

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    # 카메라 보정값 불러오기
    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3] # 카메라 내부 파라미터
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r
    # 양쪽 동일한 카메라를 사용했기 때문에 동일 행렬값을 사용

    # 이미지 불러오기
    @staticmethod
    def _load_images(filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        return images

    # 변환 행렬 생성
    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R # Rotation matrix
        T[:3, 3] = t # translation matrix
        return T

    # IMU 데이터 불러오기
    @staticmethod
    def _load_data(filepath):
        IMU_data = []
        with open(filepath, 'r') as f:
            for line in f:
                data = np.fromstring(line, dtype=np.float64, sep=' ')
                IMU_data.append(data)
        return np.array(IMU_data) # shape : (x,1) 형태 <-- 2D

    # 재투영 잔차 계산
    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    # 타일링된 키포인트 얻기 : 최적의 키 포인트 10개만 검출
    def get_tiled_keypoints(self, img, tile_h, tile_w):
        def get_kps(x, y):
            # Get the image tile
            impatch = img[y:y + tile_h, x:x + tile_w]

            # Detect keypoints
            keypoints = self.fastFeatures.detect(impatch)

            # Correct the coordinate for the point
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            # Get the 10 best keypoints
            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                return keypoints[:10]
            return keypoints

        # Get the image height and width
        h, w, *_ = img.shape

        # Get the keypoints for each of the tiles
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        # Flatten the keypoint list
        kp_list_flatten = np.concatenate(kp_list)
        return kp_list_flatten

    # 두 연속적인 이미지 간의 키포인트를 추적함
    def track_keypoints(self, img1, img2, kp1, max_error=4):
        # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2

    # 왼쪽 이미지에서 추출된 특징점에 대한 오른쪽 이미지상의 대응점을 계산함
    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)

        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)

        # Combine the masks
        in_bounds = np.logical_and(mask1, mask2)

        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]

        # Calculate the right feature points
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2

        return q1_l, q1_r, q2_l, q2_r

    # 스테레오 이미지로부터 3D 포인트를 삼각측량 방식을 계산함
    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        # Triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        # Un-homogenize
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])
        print(Q1.shape, Q2.shape)
        return Q1, Q2

    # 두 이미지에서 추출된 특징점과 3D 포인트들을 사용하여 카메라의 자세를 추정함
    def estimate_pose(self, q1, q2, Q1, Q2, IMU_yaw, max_iter=100):
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix

        # Calculate visual yaw
        visual_yaw = np.arctan2(R[1,0],R[0,0])

        # Sensor fusion
        IMU_yaw_radians = IMU_yaw * np.pi / 180
        alpha = 0.5 # alpha값 조정
        fusion_yaw = alpha*IMU_yaw_radians+(1-alpha)*visual_yaw
        print(fusion_yaw)
        # fusion rotation matrix
        fusion_R = np.copy(R)
        fusion_R[0,0] = np.cos(fusion_yaw)
        fusion_R[0,1] = -np.sin(fusion_yaw)
        fusion_R[1,0] = np.sin(fusion_yaw)
        fusion_R[1,1] = np.cos(fusion_yaw)

        transformation_matrix = self._form_transf(R, t)
        fusion_transformation_matrix = self._form_transf(fusion_R,t)
        return transformation_matrix, fusion_transformation_matrix

    # 주어진 프레임 인덱스 i에 대해 카메라의 변환 행렬을 계산함
    def get_pose(self, i):
        # Get the i-1'th image and i'th image
        img1_l, img2_l = self.images_l[i - 1:i + 1]

        # Get the IMU yaw
        IMU_yaw = self.IMU[i-1]

        # Get teh tiled keypoints
        kp1_l = self.get_tiled_keypoints(img1_l, 10, 20)

        # Track the keypoints
        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        # Calculate the disparitie
        self.disparities.append(np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16))

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])

        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Estimate the transformation matrix
        transformation_matrix,fusion_transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2, IMU_yaw)
        return transformation_matrix, fusion_transformation_matrix

def main():
    data_dir = 'straight_sequence'  # 데이터 디렉토리
    vo = VisualOdometry(data_dir)

    play_trip(vo.images_l, vo.images_r)  # Stereo 이미지 시퀀스 재생

    fusion_estimated_path = []
    estimated_path = []
    cur_pose = np.eye(4)  # 초기 pose (단위 행렬)
    cur_fusion_pose = np.eye(4)

    for i in tqdm(range(1, len(vo.images_l)), unit="frames"):
        transf, fusion_transf = vo.get_pose(i)  # i 번째 카메라 pose 계산
        # 일반 경로 업데이트
        cur_pose = np.matmul(cur_pose, transf)  # 현재 pose를 누적 계산
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))  # x, z 좌표만 추출
        # Fusion 경로 업데이트
        cur_fusion_pose = np.matmul(cur_fusion_pose, fusion_transf)
        fusion_estimated_path.append((cur_fusion_pose[0, 3], cur_fusion_pose[2, 3]))

    # 추정 경로만 시각화
    plotting.visualize_paths(estimated_path, fusion_estimated_path, "Stereo Visual Inertia Odometry",
                             file_out=os.path.basename(data_dir) + "_vo.html")

if __name__ == "__main__":
    main()
