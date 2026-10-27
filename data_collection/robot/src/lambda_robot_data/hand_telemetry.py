"""Read-only ROS 2 observer for Revo2 feedback and published STARK targets.

Runs on the G1 ROS Python environment. No ROS command publishers are created.
Timestamps mean callback receipt on G1, not motor acquisition or actuation ACK.
"""

import argparse
import math
import time

import msgpack
import zmq

JOINT_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")


def command_values(names, positions):
    if list(names) != list(JOINT_NAMES):
        raise ValueError(
            "hand JointState.name must match the documented six-joint order"
        )
    values = [float(v) for v in positions]
    if len(values) != 6 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("STARK targets must be six finite values in [0, 1]")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18560)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from ros2_stark_interfaces.msg import MotorStatus

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind("tcp://127.0.0.1:%d" % args.port)
    rclpy.init()

    class Observer(Node):
        def __init__(self):
            super().__init__("openlambda_hand_telemetry")
            self.sequence = 0
            self.commands = {}
            self.subscriptions_owned = []
            for side, state_topic in (
                ("left", "/motor_status"),
                ("right", "/motor_status_r"),
            ):
                self.subscriptions_owned.append(
                    self.create_subscription(
                        MotorStatus,
                        state_topic,
                        lambda msg, side=side: self.state(side, msg),
                        qos_profile_sensor_data,
                    )
                )
                self.subscriptions_owned.append(
                    self.create_subscription(
                        JointState,
                        "/joint_commands_" + side,
                        lambda msg, side=side: self.command(side, msg),
                        qos_profile_sensor_data,
                    )
                )

        def send(self, topic, values, mono, wall):
            self.sequence += 1
            values.update(sequence=self.sequence, monotonic_ns=mono, wall_time_ns=wall)
            publisher.send_multipart(
                [topic.encode(), msgpack.packb(values, use_bin_type=True)]
            )

        def state(self, side, msg):
            self.send(
                "rt/brainco/" + side + "/state",
                {
                    key: [int(v) for v in getattr(msg, key)]
                    for key in ("positions", "speeds", "currents", "states")
                },
                time.monotonic_ns(),
                time.time_ns(),
            )

        def command(self, side, msg):
            # Reject unknown order/units instead of silently relabeling targets.
            values = command_values(msg.name, msg.position)
            mono, wall = time.monotonic_ns(), time.time_ns()
            self.commands[side] = (mono, values)
            if len(self.commands) != 2:
                return
            if any(mono - item[0] > 50_000_000 for item in self.commands.values()):
                return
            self.send(
                "rt/brainco/glove/applied",
                {side: self.commands[side][1] for side in ("left", "right")},
                mono,
                wall,
            )

    node = Observer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        publisher.close()
        context.term()


if __name__ == "__main__":
    main()
