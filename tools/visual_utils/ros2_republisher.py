import rclpy
from rclpy.node import Node
from autoware_auto_perception_msgs.msg import DetectedObjects
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.timer import Timer

class DetectedObjectsRepublisher(Node):

    def __init__(self):
        super().__init__('detected_objects_republisher')

        # Define QoS profile to match the publisher
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)

        # Create publishers for republishing messages
        self.republished_det_debug_publisher = self.create_publisher(
            DetectedObjects, 'detected_objects', qos_profile)
        self.republished_ground_truth_publisher = self.create_publisher(
            DetectedObjects, 'ground_truth', qos_profile)

        # Create subscribers for both topics
        self.subscription_debug = self.create_subscription(
            DetectedObjects,
            'detected_objects_debug',
            self.detected_objects_debug_callback,
            qos_profile)

        self.subscription_ground_truth = self.create_subscription(
            DetectedObjects,
            'ground_truth_objects',
            self.ground_truth_objects_callback,
            qos_profile)

        # Initialize variables to store the latest received messages
        self.latest_detected_objects_debug = None
        self.latest_ground_truth_objects = None

        # Timer callback to publish at 10 Hz (0.1 seconds)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def detected_objects_debug_callback(self, msg: DetectedObjects):
        # Callback to handle messages from 'detected_objects_debug'
        self.latest_detected_objects_debug = msg

    def ground_truth_objects_callback(self, msg: DetectedObjects):
        # Callback to handle messages from 'ground_truth_objects'
        self.latest_ground_truth_objects = msg

    def timer_callback(self):
        # Publish the latest received 'detected_objects_debug' message if available
        if self.latest_detected_objects_debug:
            self.republished_det_debug_publisher.publish(self.latest_detected_objects_debug)

        # Publish the latest received 'ground_truth_objects' message if available
        if self.latest_ground_truth_objects:
            self.republished_ground_truth_publisher.publish(self.latest_ground_truth_objects)

def main(args=None):
    rclpy.init(args=args)
    node = DetectedObjectsRepublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

