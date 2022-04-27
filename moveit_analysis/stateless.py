
from typing import Optional,Tuple,Sequence
import sys

import rclpy
from rclpy.duration import Duration

import PyKDL as kdl

import moveit_analysis.convert as convert

from std_msgs.msg import Header
from geometry_msgs.msg import Pose

from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from moveit_msgs.msg import (
        Constraints,
        PositionConstraint,
        OrientationConstraint,
        RobotState,
        RobotTrajectory,
        PositionIKRequest,
        MotionPlanRequest,
        JointConstraint,
        BoundingVolume,
        )

from moveit_msgs.srv import (
        GetMotionPlan,
        GetPositionFK,
        GetPositionIK,
        GetCartesianPath
        )

# Factories
def joint_state_factory(group, positions):
    """
    Get a JointState message from a group and a set of ordered positions.
    """
    return JointState(
            header=Header(),
            name=group.get_active_joints(),
            position=positions)

def joint_map_factory(group, positions):
    """
    Get a JointState message from a group and a set of ordered positions.
    """
    return dict(zip(
            group.get_active_joints(),
            positions))

def joint_map_to_joint_state(joint_map):
    joint_state = JointState()
    if joint_map is not None and len(joint_map) > 0:
        joint_state.name, joint_state.position = zip(*joint_map.items())
    return joint_state

def joint_state_to_joint_map(joint_state):
    return dict(zip(joint_state.name, joint_state.position))

def union(*dicts):
    """Recursively construct a union of dictionaries"""

    return reduce(lambda x,y: dict(x, **y), dicts, {})

def robot_state_factory(joints):
    """Get MoveIt robot state for a given robot posture."""

    if type(joints) is JointState:
        joint_state = joints
    elif type(joints) is dict:
        joint_state = joint_map_to_joint_state(joints)

    robot_state = RobotState()
    robot_state.joint_state = joint_state

    return robot_state

def cart_constraints(name, stamp, link_name, frame_id, pose, pos_tolerance=None, rot_tolerance=None):
    """
    helper to populate a cartesian constraint set
    """
    constraints = Constraints(name=name)
    if pos_tolerance is not None:
        constraints.position_constraints = [
                PositionConstraint(
                    header                    = Header(
                        stamp                 = stamp,
                        frame_id              = frame_id
                        ),
                    link_name                 = link_name,
                    constraint_region         = BoundingVolume(
                        primitives            = [SolidPrimitive(type=2, dimensions=[pos_tolerance])],
                        primitive_poses       = [pose]
                        ),
                    weight                    = 1.0
                    )
                ]

    if rot_tolerance is not None:
        constraints.orientation_constraints = [
                OrientationConstraint(
                    header                    = Header(
                        stamp                 = stamp,
                        frame_id              = frame_id,
                        ),
                    link_name                 = link_name,
                    orientation               = pose.orientation,
                    parameterization          = 0,
                    absolute_x_axis_tolerance = rot_tolerance[0],
                    absolute_y_axis_tolerance = rot_tolerance[1],
                    absolute_z_axis_tolerance = rot_tolerance[2],
                    weight                    = 1.0)
                ]

    return constraints

class PlannerInterface(object):
    """
    Stateless interface to a moveit planner
    """
    def __init__(self, node):

        # store node so that we can tell the time
        self.node = node

        # create service clients
        self._ik        = node.create_client(GetPositionIK,    'compute_ik')
        self._fk        = node.create_client(GetPositionFK,    'compute_fk')
        self._plan      = node.create_client(GetMotionPlan,    'plan_kinematic_path')
        self._plan_cart = node.create_client(GetCartesianPath, 'compute_cartesian_path')

        # wait for services
        for cli in [self._ik, self._fk, self._plan, self._plan_cart]:
            while not cli.wait_for_service(timeout_sec=1.0):
                node.get_logger().warn(f'waiting for service: {cli.srv_name}')
            node.get_logger().info(f'connected to service: {cli.srv_name}')


    def get_transform(self,
            robot_state:  RobotState,
            common_frame: str,
            from_frame:   str,
            to_frame:     str
            ) -> Optional[kdl.Frame]:
        """
        Get KDL transform from from_frame to to_frame based on a given robot
        posture.

        The calculation is done in the common_frame.
        """

        req = GetPositionFK.Request()
        req.header.stamp = self.node.get_clock().now().to_msg()
        req.header.frame_id = common_frame
        req.fk_link_names = [from_frame, to_frame]
        req.robot_state = robot_state

        res = self._fk.call(req)

        if res.error_code.val != 1:
            print('Could not get transform, error: {}'.format(res.error_code.val))
            return None
        else:
            F_from = convert.PoseMsg.To.KDL.Frame(res.pose_stamped[0].pose)
            F_to = convert.PoseMsg.To.KDL.Frame(res.pose_stamped[1].pose)

            return (F_from.Inverse() * F_to)

    def get_ik(self,
            robot_state:      RobotState,
            group_name:       str,
            link_name:        str,
            frame_id:         str,
            pose:             kdl.Frame,
            avoid_collisions: bool = False
            ) -> Optional[RobotState]:
        """
        get an ik solution that positions link_name link at the given pose
        relative to the frame_id
        """

        req = PositionIKRequest()
        req.group_name=group_name
        req.robot_state = robot_state
        req.robot_state.joint_state.header.stamp = self.node.get_clock().now().to_msg()
        req.avoid_collisions = avoid_collisions
        req.ik_link_name = link_name
        req.pose_stamped.header.frame_id = frame_id
        req.pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        req.pose_stamped.pose = convert.to_pose_msg(pose)
        # TODO: what is the replacement for this
        # req.attempts = 20
        req.timeout = Duration(seconds=0.01).to_msg()

        # self.node.get_logger().info(str(req))

        res = self._ik.call(GetPositionIK.Request(ik_request=req))

        # self.node.get_logger().info(f'ik solution: solution.joint_state.position')

        if res.error_code.val == 1:
            return res.solution
        else:
            self.node.get_logger().debug('IK solver failed, res: {}'.format(res.error_code))
            return None

    def get_path(
            self,
            start_state: RobotState,
            group_name:  str,
            link_name:   str,
            frame_id:    str,
            poses:       Sequence[kdl.Frame],
            ) -> Optional[RobotTrajectory]:

        """
        Compute a cartesian path along the specified poses, starting in the
        given joint state.
        """

        # Request cart motion plan
        req = GetCartesianPath.Request()
        req.header.frame_id=frame_id
        req.header.stamp=self.node.get_clock().now().to_msg()
        req.start_state = start_state
        req.start_state.is_diff=True
        req.waypoints=[convert.to_pose_msg(p) for p in poses]
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.group_name=group_name
        req.link_name=link_name
        req.avoid_collisions=True

        # request plan
        res = self._plan_cart.call(req)

        if res.error_code.val == 1:
            self.node.get_logger().debug('IK solution found.')
            return res.solution
        else:
            self.node.get_logger().debug('IK solver failed, res: {}'.format(res.error_code))
            return None

    def get_joint_plan(self,
            group_name,
            start,
            end):

        req = MotionPlanRequest()
        req.group_name=group_name
        req.start_state=robot_state_factory(start)
        req.start_state.is_diff = True
        req.num_planning_attempts=1
        req.allowed_planning_time=10.0

        req.goal_constraints.append(Constraints())
        req.goal_constraints[0].name='goal'

        for name, val in end.items():
            req.goal_constraints[0].joint_constraints.append(JointConstraint())
            req.goal_constraints[0].joint_constraints[-1].joint_name = name
            req.goal_constraints[0].joint_constraints[-1].position = val
            req.goal_constraints[0].joint_constraints[-1].tolerance_above = 0.01
            req.goal_constraints[0].joint_constraints[-1].tolerance_below = 0.01
            req.goal_constraints[0].joint_constraints[-1].weight = 1.0

        res = self._plan.call(req).motion_plan_response

        if res.error_code.val == 1:
            rospy.loginfo('Plan found.')
            return res
        else:
            rospy.logerr('Planner failed, res: {}'.format(res.error_code))
            return None

    def get_plan(self,
            group_name,
            frame_id,
            pose,
            link_name,
            joints):

        pose_msg = convert.KDL.Frame.To.PoseMsg(pose)

        req = MotionPlanRequest()
        req.group_name=group_name
        req.start_state=robot_state_factory(joints)
        req.start_state.is_diff = True
        req.num_planning_attempts=1
        req.allowed_planning_time=10.0

        req.goal_constraints.append(Constraints())
        req.goal_constraints[0].name='goal'

        req.goal_constraints[0].position_constraints.append(PositionConstraint())
        req.goal_constraints[0].position_constraints[0].header.frame_id=frame_id
        req.goal_constraints[0].position_constraints[0].header.stamp=rospy.Time.now()
        req.goal_constraints[0].position_constraints[0].link_name=link_name
        req.goal_constraints[0].position_constraints[0].constraint_region.primitives.append(SolidPrimitive())
        req.goal_constraints[0].position_constraints[0].constraint_region.primitives[0].type=2
        req.goal_constraints[0].position_constraints[0].constraint_region.primitives[0].dimensions=[0.001]
        req.goal_constraints[0].position_constraints[0].constraint_region.primitive_poses.append(pose_msg)

        req.goal_constraints[0].orientation_constraints.append(OrientationConstraint())
        req.goal_constraints[0].orientation_constraints[0].header.frame_id=frame_id
        req.goal_constraints[0].orientation_constraints[0].header.stamp=rospy.Time.now()
        req.goal_constraints[0].orientation_constraints[0].link_name=link_name
        req.goal_constraints[0].orientation_constraints[0].orientation=pose_msg.orientation
        req.goal_constraints[0].orientation_constraints[0].absolute_x_axis_tolerance=0.01
        req.goal_constraints[0].orientation_constraints[0].absolute_y_axis_tolerance=0.01
        req.goal_constraints[0].orientation_constraints[0].absolute_z_axis_tolerance=0.01

        res = self._plan.call(req).motion_plan_response

        if res.error_code.val == 1:
            rospy.loginfo('Plan found.')
            return res
        else:
            rospy.logerr('Planner failed, res: {}'.format(res.error_code))
            return None

    def get_approach_plan(self,
            start_state: RobotState,
            group_name:  str,
            link_name:   str,
            frame_id:    str,
            start_pose:  kdl.Frame,
            end_pose:    kdl.Frame,
            goal_tolerance: float,
            orientation_tolerance: Tuple[float,float,float]):
        """
        goal_tolerance: spherical
        orientation_tolerance: r,p,y angles in frame_id

        compute two motion plans:
        1. plan from current state to end_pose
        2. plan from end_pose to start_pose withgiven orientation tolerance
        """

        # convert to messae etypes
        start_pose_msg = convert.KDL.Frame.To.PoseMsg(start_pose)
        end_pose_msg = convert.KDL.Frame.To.PoseMsg(end_pose)

        now = self.node.get_clock().now().to_msg()

        start_state.is_diff = True

        end_req = MotionPlanRequest(
                group_name            = group_name,
                start_state           = start_state,
                num_planning_attempts = 3,
                allowed_planning_time = 30.0,
                goal_constraints      = [cart_constraints(
                    'goal',
                    now,
                    link_name,
                    frame_id,
                    end_pose_msg,
                    goal_tolerance,
                    orientation_tolerance)]
                )

        end_res = self._plan.call(GetMotionPlan.Request(motion_plan_request=end_req)).motion_plan_response

        if end_res.error_code.val != 1:
            return None

        approach_req = MotionPlanRequest(
                group_name            = group_name,
                start_state           = RobotState(
                    joint_state       = JointState(
                        name          = end_res.trajectory.joint_trajectory.joint_names,
                        position      = end_res.trajectory.joint_trajectory.points[-1].positions
                        ),
                    is_diff           = True
                    ),
                num_planning_attempts = 3,
                allowed_planning_time = 30.0,
                goal_constraints      = [cart_constraints(
                    'goal',
                    now,
                    link_name,
                    frame_id,
                    start_pose_msg,
                    goal_tolerance,
                    orientation_tolerance)],
                path_constraints = cart_constraints(
                    'coaxial',
                    now,
                    link_name,
                    frame_id,
                    start_pose_msg,
                    pos_tolerance=None,
                    rot_tolerance=orientation_tolerance
                    )
                )

        approach_res = self._plan.call(GetMotionPlan.Request(motion_plan_request=approach_req)).motion_plan_response

        if approach_res.error_code.val != 1:
            return None

        return approach_res

    def get_cart_approach_plan(self,
            start_state: RobotState,
            group_name:  str,
            link_name:   str,
            frame_id:    str,
            start_pose:  kdl.Frame,
            end_pose:    kdl.Frame,
            goal_tolerance: float,
            orientation_tolerance: Tuple[float,float,float]):
        """
        goal_tolerance: spherical
        orientation_tolerance: r,p,y angles in frame_id

        compute two motion plans:
        1. plan from current state to end_pose
        2. plan from end_pose to start_pose withgiven orientation tolerance
        """

        # convert to messae etypes
        start_pose_msg = convert.KDL.Frame.To.PoseMsg(start_pose)
        end_pose_msg = convert.KDL.Frame.To.PoseMsg(end_pose)

        now = self.node.get_clock().now().to_msg()

        start_state.is_diff = True


        end_req = MotionPlanRequest(
                group_name            = group_name,
                start_state           = start_state,
                num_planning_attempts = 2,
                allowed_planning_time = 10.0,
                goal_constraints      = [cart_constraints(
                    'goal',
                    now,
                    link_name,
                    frame_id,
                    end_pose_msg,
                    pos_tolerance=goal_tolerance,
                    rot_tolerance=orientation_tolerance)]
                )

        end_res = self._plan.call(GetMotionPlan.Request(motion_plan_request=end_req)).motion_plan_response

        if end_res.error_code.val != 1:
            return None

        # Request cart motion plan
        req = GetCartesianPath.Request()
        req.header.frame_id=frame_id
        req.header.stamp=self.node.get_clock().now().to_msg()
        req.start_state = RobotState(
                    joint_state       = JointState(
                        name          = end_res.trajectory.joint_trajectory.joint_names,
                        position      = end_res.trajectory.joint_trajectory.points[-1].positions
                        ),
                    is_diff           = True
                    )
        req.start_state.is_diff=True
        req.waypoints=[end_pose_msg, start_pose_msg]
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.group_name=group_name
        req.link_name=link_name
        req.avoid_collisions=True
        req.path_constraints = cart_constraints(
                    'coaxial',
                    now,
                    link_name,
                    frame_id,
                    start_pose_msg,
                    pos_tolerance=None,
                    rot_tolerance=orientation_tolerance
                )

        # request plan
        approach_res = self._plan_cart.call(req)

        if approach_res.error_code.val != 1:
            return None

        return approach_res.solution
