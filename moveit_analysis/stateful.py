
import os
import copy

import rclpy

import PyKDL as kdl

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import (
        CollisionObject,
        AttachedCollisionObject,
        PlanningScene,
        RobotState,
        LinkScale,
        LinkPadding
        )

# only required for generating meshes
# from moveit_commander.planning_scene_interface import PlanningSceneInterface

import moveit_analysis.convert as convert

class WorldInterface(object):
    def __init__(self, node, ns=None):
        """
        Interface to the moveit planning scene, which maintains world state.

        node: for ros interfaces
        ns: namespace for joint states
        """

        self.node = node

        joint_states_topic = 'joint_states'
        if ns:
            joint_states_topic = '/'.join([ns, joint_states_topic])

        self.js_pub = node.create_publisher(
                JointState,
                joint_states_topic,
                1)

        self.set_scene = node.create_client(
                ApplyPlanningScene,
                'apply_planning_scene')

        while not self.set_scene.wait_for_service(timeout_sec=1.0):
            node.get_logger().warn(f'waiting for service: {self.set_scene.srv_name}')
        node.get_logger().info(f'connected to service: {self.set_scene.srv_name}')

    def clear_world(self):
        """Clear all scene information, including allowed collisions."""
        # construct scene message
        scene_msg = PlanningScene()
        scene_msg.is_diff=False
        scene_msg.robot_state.is_diff=True

        # set the scene synchronously
        self.set_scene.call(ApplyPlanningScene.Request(scene=scene_msg))

    def set_joint_state(self, robot_state: RobotState):
        """
        Set the joint state for the robot
        """

        # construct joint state message
        js = JointState()
        js.name     = list(robot_state.joint_state.name)
        js.position = list(robot_state.joint_state.position)

        js.header.stamp = self.node.get_clock().now().to_msg()
        self.js_pub.publish(js)

        # construct scene message
        scene_msg = PlanningScene()
        scene_msg.is_diff=True
        scene_msg.robot_state.is_diff=True
        scene_msg.robot_state.joint_state=js

        # set the scene synchronously
        self.set_scene.call(ApplyPlanningScene.Request(scene=scene_msg))

