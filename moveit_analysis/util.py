
from typing import Sequence,Tuple,Callable
from types import SimpleNamespace

import threading

import rclpy.node
import rclpy.qos
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy import Parameter

import std_msgs.msg as std_msgs


def embed_parameters(
        node:       rclpy.node.Node,
        parameters: Sequence[Tuple[str, Parameter.Type]]
        ) -> None:
    """
    Declare and get a list of parameters and assign them as members of the
    given node object. For nested parameters, create `SimpleNamespace` objects
    to hold them.
    """
    for name, param_type in parameters:

        node.declare_parameter(name, param_type)

        leaf = node
        tokens = name.split('.')
        while len(tokens) > 1:
            token = tokens.pop(0)
            if not hasattr(leaf, token):
                setattr(leaf, token, SimpleNamespace())
            leaf = getattr(leaf, token)

        setattr(leaf, tokens[0], node.get_parameter(name))

latching_qos = QoSProfile(
        depth       = 1,
        reliability = rclpy.qos.ReliabilityPolicy.RELIABLE,
        durability  = rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
        history     = rclpy.qos.HistoryPolicy.KEEP_LAST)


class ModelListener:
    """
    Utility class that calls a callback when a
    robot model is received.
    """
    def __init__(self,
            node: rclpy.node.Node,
            callback: Callable[[rclpy.node.Node, str], None]):

        self.node = node
        self.callback = callback

        # TODO: avoid needing thread
        self.process_thread = None

        # subscribe to robot description topic
        self.urdf_sub = self.node.create_subscription(
                std_msgs.String,
                'robot_description',
                self.robot_description_cb,
                qos_profile = latching_qos)

    def robot_description_cb(self, urdf_msg):
        """
        spawn thread to handle robot description and analysis
        """

        self.process_thread = threading.Thread(
                target=self.callback,
                args=[self.node, urdf_msg.data])

        self.process_thread.start()
