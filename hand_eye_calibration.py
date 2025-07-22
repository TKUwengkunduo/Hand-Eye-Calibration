import os
import json
import numpy as np
import cv2
from units.charuco_detector import ChArUcoDetector


class HandEyeCalibrator:
    def __init__(self, data_path, camera_matrix, dist_coeffs):
        self.data_path = data_path
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.detector = ChArUcoDetector(camera_matrix, dist_coeffs)

        # 儲存轉換矩陣
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []

    @staticmethod
    def rvec_tvec_to_matrix(rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        return T

    @staticmethod
    def quaternion_to_matrix(x, y, z, w):
        # 正規化
        n = x*x + y*y + z*z + w*w
        if n < np.finfo(float).eps:
            return np.eye(3)
        x, y, z, w = x/np.sqrt(n), y/np.sqrt(n), z/np.sqrt(n), w/np.sqrt(n)
        
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])


    def pose_to_matrix(self, pose):
        R = self.quaternion_to_matrix(
            pose["orientation"]["x"],
            pose["orientation"]["y"],
            pose["orientation"]["z"],
            pose["orientation"]["w"]
        )
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [
            pose["position"]["x"],
            pose["position"]["y"],
            pose["position"]["z"]
        ]
        return T

    def load_data(self):
        with open(self.data_path, "r") as f:
            dataset = json.load(f)
        return dataset

    def process_data(self):
        dataset = self.load_data()
        for entry in dataset:
            image_path = entry["image_path"]
            pose = entry["pose"]

            T_be = self.pose_to_matrix(pose)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[Warning] 無法讀取圖片: {image_path}")
                continue

            rvec, tvec, _ = self.detector.detect(image, draw=False)
            if rvec is None or tvec is None:
                print(f"[Warning] 無法偵測 ChArUco: {image_path}")
                continue

            T_cg = self.rvec_tvec_to_matrix(rvec, tvec)
            self.R_gripper2base.append(T_be[:3, :3])
            self.t_gripper2base.append(T_be[:3, 3])
            self.R_target2cam.append(T_cg[:3, :3])
            self.t_target2cam.append(T_cg[:3, 3])

    def calibrate(self):
        if len(self.R_gripper2base) < 2:
            raise RuntimeError("有效資料不足，無法進行手眼校正。")
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            self.R_gripper2base, self.t_gripper2base,
            self.R_target2cam, self.t_target2cam
        )
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.ravel()
        return T_cam2gripper

    def run(self):
        self.process_data()
        T = self.calibrate()
        print("\n=== 手眼校正結果 (End->Camera) ===")
        print("Homogeneous Matrix (T):\n", T)


if __name__ == "__main__":
    camera_matrix = np.array([
        [905.1003425766808, 0.0, 637.2918791234757],
        [0.0, 903.7173217099158, 361.4302168462347],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([
        0.14464552813304468,
        -0.3203500793804207,
        0.0006902606917917997,
        -0.0005571665140519677,
        0.0005571665140519677
    ])
    data_path = os.path.join("hand_eye_data", "data.json")

    calibrator = HandEyeCalibrator(data_path, camera_matrix, dist_coeffs)
    calibrator.run()
