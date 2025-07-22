import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import json
import os
from datetime import datetime

class HandEyeDataCollector(Node):
    def __init__(self):
        super().__init__('hand_eye_data_collector')

        # 訂閱機械手末端 Pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/robot/end_effector_pose',  # 可根據實際topic修改
            self.pose_callback,
            10
        )

        # 訂閱相機影像
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',  # 可根據實際topic修改
            self.image_callback,
            10
        )

        self.current_pose = None
        self.current_image = None
        self.bridge = CvBridge()

        # 資料儲存目錄
        self.data_dir = os.path.join(os.getcwd(), 'hand_eye_data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.json_path = os.path.join(self.data_dir, 'data.json')
        self.data_records = []

        # 嘗試讀取現有JSON
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                self.data_records = json.load(f)

        self.get_logger().info('Hand-eye data collector started.')

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = {
            'position': {
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z
            },
            'orientation': {
                'x': msg.pose.orientation.x,
                'y': msg.pose.orientation.y,
                'z': msg.pose.orientation.z,
                'w': msg.pose.orientation.w
            }
        }

    def image_callback(self, msg: Image):
        self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow("Camera View", self.current_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and self.current_pose is not None:
            self.save_data()

    def save_data(self):
        # 使用timestamp生成檔名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        img_filename = f'image_{timestamp}.png'
        img_path = os.path.join(self.data_dir, img_filename)
        cv2.imwrite(img_path, self.current_image)

        record = {
            'image_path': img_path,
            'pose': self.current_pose
        }
        self.data_records.append(record)

        with open(self.json_path, 'w') as f:
            json.dump(self.data_records, f, indent=4)

        self.get_logger().info(f'Data saved: {img_filename}')

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeDataCollector()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down.')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
