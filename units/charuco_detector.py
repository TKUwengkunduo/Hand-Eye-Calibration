"""========================================
opencv-contrib-python   4.7.0.72
opencv-python           4.12.0.88
numpy                   1.24.1
========================================"""

import cv2
import numpy as np


class ChArUcoDetector:
    def __init__(self, camera_matrix, dist_coeffs, squares_x=10, squares_y=14, square_length=0.02, marker_length=0.015,
                 dictionary=cv2.aruco.DICT_4X4_100):
        """
        初始化 ChArUco 偵測器
        :param squares_x: 棋盤格的列數
        :param squares_y: 棋盤格的行數
        :param square_length: 每個棋盤格邊長 (公尺)
        :param marker_length: ArUco 標記邊長 (公尺)
        :param dictionary: ArUco 字典類型
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length

        # 初始化字典與棋盤
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
                self.square_length,
                self.marker_length,
                self.aruco_dict)
        self.params = cv2.aruco.DetectorParameters()

    def detect(self, image, draw=True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.params
        )

        rvec, tvec = None, None
        if ids is not None and len(ids) > 0:
            num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board
            )
            if num_corners > 0:
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self.board,
                    self.camera_matrix, self.dist_coeffs, None, None
                )

                if draw:
                    cv2.aruco.drawDetectedMarkers(image, corners, ids)
                    cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
                    if retval:
                        cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, length=0.1, thickness=3)
        
        return rvec, tvec, image



def test_image(image_path, camera_matrix, dist_coeffs):
    
    detector = ChArUcoDetector(camera_matrix, dist_coeffs)

    image = cv2.imread(image_path)
    rvec, tvec, result_img = detector.detect(image)

    print("rvec = ", rvec.ravel())
    print("tvec = ", tvec.ravel())

    cv2.imshow("ChArUco Detection", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def test_camera(camera_id=0):
    detector = ChArUcoDetector()
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("無法開啟攝影機")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rvec, tvec, result_img = detector.detect(frame)

        print("rvec = ", rvec.ravel())
        print("tvec = ", tvec.ravel())
        cv2.imshow("ChArUco Detection (Camera)", result_img)

        key = cv2.waitKey(1)
        if key == 27:  # ESC 離開
            break

    cap.release()
    cv2.destroyAllWindows()


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


    # 測試照片
    test_image("image/aruco_Color.png", camera_matrix, dist_coeffs)

    # 測試攝影機
    # test_camera(0)
