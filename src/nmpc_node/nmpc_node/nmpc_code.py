import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
import numpy as np
import do_mpc
from casadi import *

class NMPCNode(Node):
    def __init__(self):
        super().__init__("nmpc_node")

        self.dt = 0.1  # Time step
        self.N = 20  # Prediction horizon

        self.x0 = np.zeros((3, 1))  # Initial state [x, y, theta]
        self.u0 = np.zeros((2, self.N))  # Initial control [v, omega]
        self.x_ref = np.zeros((3, 1))  # Reference position [x, y, theta]

        # Define model and NMPC
        self.model = self.define_model()
        self.mpc = self.define_mpc()

        # ROS 2 Publishers and Subscribers
        self.control_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_subscription = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.cone_subscription = self.create_subscription(Point, "/cone_position", self.cone_callback, 10)

        # Timer to run NMPC loop
        self.timer = self.create_timer(self.dt, self.nmpc_callback)

    def define_model(self):
        model_type = do_mpc.model.Model("continuous")

        x = model_type.set_variable(var_type="state", var_name="x", shape=(1,))
        y = model_type.set_variable(var_type="state", var_name="y", shape=(1,))
        theta = model_type.set_variable(var_type="state", var_name="theta", shape=(1,))

        v = model_type.set_variable(var_type="input", var_name="v", shape=(1,))
        omega = model_type.set_variable(var_type="input", var_name="omega", shape=(1,))

        model_type.set_rhs("x", v * cos(theta))
        model_type.set_rhs("y", v * sin(theta))
        model_type.set_rhs("theta", omega)

        model_type.setup()
        return model_type

    def define_mpc(self):
        mpc = do_mpc.controller.MPC(self.model)

        setup_mpc = {
            'n_horizon': self.N,
            't_step': self.dt,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'store_full_solution': True,
            'nlpsol_opts': {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        }
        mpc.set_param(**setup_mpc)

        mpc.set_objective(
            mterm=((self.model.x['x'] - self.x_ref[0])**2 + (self.model.x['y'] - self.x_ref[1])**2),
            lterm=((self.model.x['x'] - self.x_ref[0])**2 + (self.model.x['y'] - self.x_ref[1])**2 +
                   self.model.u['v']**2 + self.model.u['omega']**2)
        )

        mpc.bounds['lower', 'u', 'v'] = 0.0
        mpc.bounds['upper', 'u', 'v'] = 1.5
        mpc.bounds['lower', 'u', 'omega'] = -1.0
        mpc.bounds['upper', 'u', 'omega'] = 1.0

        mpc.setup()
        return mpc

    def odom_callback(self, msg):
        self.x0[0] = msg.pose.pose.position.x
        self.x0[1] = msg.pose.pose.position.y
        self.x0[2] = msg.pose.pose.orientation.z

    def cone_callback(self, msg):
        self.x_ref[0] = msg.x
        self.x_ref[1] = msg.y

    def nmpc_callback(self):
        try:
            u0 = self.mpc.make_step(self.x0)

            cmd = Twist()
            cmd.linear.x = float(u0[0])
            cmd.angular.z = float(u0[1])
            self.control_publisher.publish(cmd)
        except Exception as e:
            self.get_logger().error(f"Error in NMPC computation: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = NMPCNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
