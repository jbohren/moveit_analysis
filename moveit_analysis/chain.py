
from typing import Sequence

import sys
import os
import threading
import time
import pickle

from types import SimpleNamespace

from math import sqrt

import numpy as np
np.set_printoptions(suppress=False)
np.set_printoptions(precision=1)

import PyKDL as kdl
import kdl_parser_py.urdf

import moveit_analysis.convert as convert
from moveit_analysis.stateless import PlannerInterface
from moveit_analysis.stateful import WorldInterface
from moveit_analysis.util import latching_qos

from moveit_msgs.msg import (
        Constraints,
        PositionConstraint,
        OrientationConstraint,
        RobotState,
        RobotTrajectory,
        PositionIKRequest,
        MotionPlanRequest,
        JointConstraint,
        )


class ChainModel:
    """
    Model of a serial chain manipulator with appropriate solvers.
    """

    def __init__(self,
            urdf_str,
            base_link,
            tip_link,
            planning_group):

        # Store parameters
        self.urdf_str  = urdf_str
        self.base_link = base_link
        self.tip_link  = tip_link
        self.planning_group = planning_group

        # Construct KDL robot model and solvers
        (_, self.tree)  = kdl_parser_py.urdf.treeFromString(urdf_str)
        self.chain      = self.tree.getChain(self.base_link, self.tip_link)

        self.fk_solver  = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.ik_vel     = kdl.ChainIkSolverVel_pinv(self.chain)

        # Get non-fixed joint names
        self.joint_names = [
                joint.getName()
                for joint
                in [self.chain.getSegment(s).getJoint() for s in range(0, self.chain.getNrOfSegments())]
                if joint.getType() != kdl.Joint.JointType.Fixed
                ]
        self.joint_index_map = {
                    name: index
                    for index, name
                    in enumerate(self.joint_names)
                }
        self.joint_types = [
                joint.getType()
                for joint
                in [self.chain.getSegment(s).getJoint() for s in range(0, self.chain.getNrOfSegments())]
                if joint.getType() != kdl.Joint.JointType.Fixed
                ]
        print(self.joint_names)
        print(self.joint_types)
        self.n_joints = len(self.joint_names)

    def get_posture(
            self,
            robot_state: RobotState
            ) -> Sequence[float]:
        """
        Get a list of joint positions from a moveit RobotState message.
        """

        return [robot_state.joint_state.position[self.joint_index_map[name]]
                for name
                in self.joint_names
                if name in robot_state.joint_state.name]

    def get_robot_state(
            self,
            posture: Sequence[float]
            ) -> RobotState:
        """
        Get a moveit robot state message from a given list of joint positions.
        """

        robot_state = RobotState()
        robot_state.joint_state.name = list(self.joint_names)
        robot_state.joint_state.position = list(posture)

        return robot_state

def compute_metrics(
        model,
        positions,
        orientations,
        postures,
        metric_specs):
    """
    Compute a set of metrics for a robot model over a set of joint postures.

    positions: np.array of Cart positions (x,y,z)
    orientations: np.array of Cart orientations (x,y,z,w)
    postures: 3D np.ndarray of joint positions
    metric_specs: sequence of (label, functor, dict) with function signature fun(model, features, **kwargs)

    returns: 2D np array of poses, where each element is a namespace including:
        - q: posture
        - F: pose
        - J: jacobian
        - <metric>: each user-supplied metric
    """

    results = np.ndarray((len(positions), len(orientations)), dtype=object)

    # Iterate over sample space by position/orientation
    for ipos, (x,y,z) in enumerate(positions):
        for iori, (qx, qy, qz, qw) in enumerate(orientations):

            # Skip if no solution at this pose
            posture = postures[ipos, iori]
            if posture is None:
                continue

            # Construct Cartesian frame
            F = kdl.Frame(
                    kdl.Rotation.Quaternion(x=qx, y=qy, z=qz, w=qw),
                    kdl.Vector(x,y,z)
                    )

            # Precompute Jacobian
            q = convert.Array.To.KDL.JntArray(posture)
            J = kdl.Jacobian(q.rows())
            model.jac_solver.JntToJac(q, J)

            # Store in result for this pose
            metrics = SimpleNamespace()
            metrics.q = posture
            metrics.F = convert.KDL.Frame.To.Mat(F)
            metrics.J = convert.KDL.Jacobian.To.Mat(J)

            # Compute metrics
            for (label, fun, kwargs) in metric_specs:
                setattr(metrics, label, fun(model, metrics, **kwargs))

            # Set the results for this pos/orientation
            results[ipos, iori] = metrics

    return results


def joint_load(model, metrics, wrench, g_vec):
    """
    compute joint loads as a function of a tip load with some gravity

    wrench: wrench on last link in chain
    g_vec: gravity vector defined in the base frame
    """
    n = model.chain.getNrOfJoints()
    q = metrics.q

    g = kdl.Vector(g_vec[0],g_vec[1],g_vec[2])
    id_solver =  kdl.ChainIdSolver_RNE(model.chain, g)

    # Construct kdl structures
    w_ee = kdl.Wrench(
        kdl.Vector(*wrench[0:3]),
        kdl.Vector(*wrench[3:]))

    # No motion / no external loads besides EE
    q_dot = kdl.JntArray(n)
    q_dot_dot = kdl.JntArray(n)
    w0 = kdl.Wrench()

    kdl.SetToZero(q_dot)
    kdl.SetToZero(q_dot_dot)
    kdl.SetToZero(w0)

    # Construct wrench set
    wrenches = ([w0] * n) + [w_ee]

    joint_torques = kdl.JntArray(n)
    ret = id_solver.CartToJnt(q, q_dot, q_dot_dot, wrenches, joint_torques)

    torques = np.zeros(n)
    for i in range(n):
        torques[i] = joint_torques[i]

    return torques

def joint_vel(model, metrics, twist):
    """
    compute instantaneous joint velocity from given end-effector twist

    twist: tip twist
    """

    q = metrics.q
    n = model.chain.getNrOfJoints()

    t = kdl.Twist(
        kdl.Vector(*twist[0:3]),
        kdl.Vector(*twist[3:]))

    q_dot = kdl.JntArray(n)
    model.ik_vel.CartToJnt(q, t, q_dot)

    vels = np.zeros(n)
    for i in range(n):
        vels[i] = q_dot[i]

    return vels

def manipulability(model, metrics):
    """
    """

    try:
        J = metrics.J

        r = np.linalg.matrix_rank(J)
        if r == 6:
            return sqrt(np.linalg.det(J*J.T))/6.0
        else:
            return 0
    except ValueError:
        return 0

def stiffness(model, metrics, k):
    """
    compute cartesian stiffness tesnor from joint-space stiffnesses k at a given position

    Kx = (J * Kq^-1 * K^T)^-1
    """
    J = metrics.J
    cart_stiffness = (J * np.matrix(np.diag([k]*len(position))).I * J.T).I

    return cart_stiffness

def local_stiffness(model, metrics):

    # Get principal components
    l,_ = np.linalg.eig(Cxyz)
    l = sorted(l)

    metric_data['Cx_local'][ijk].append(Cxyz_local[0,0])
    metric_data['Cy_local'][ijk].append(Cxyz_local[1,1])
    metric_data['Cz_local'][ijk].append(Cxyz_local[2,2])
    metric_data['Cxyz_eig1'][ijk].append(np.min(l))
    metric_data['Cxyz_eig2'][ijk].append(np.max(l))
    metric_data['Cxyz_eig3'][ijk].append(np.max(l))

def force_resolution(model, metrics, f_min):
    load_inc_x = model.joint_load(position, np.stack([[R.T*np.matrix([f_min,0,0]).T],[np.matrix([0,0,0]).T]]).flatten(), np.array([0,0,0]))
    load_inc_y = model.joint_load(position, np.stack([[R.T*np.matrix([0,f_min,0]).T],[np.matrix([0,0,0]).T]]).flatten(), np.array([0,0,0]))
    load_inc_z = model.joint_load(position, np.stack([[R.T*np.matrix([0,0,f_min]).T],[np.matrix([0,0,0]).T]]).flatten(), np.array([0,0,0]))

    # load_inc = [
        # min([abs(v) for v in (x,y,z,1e+6) if abs(v) > 1E-6])
        # for x,y,z
        # in zip(load_inc_x[0:6], load_inc_y[0:6], load_inc_z[0:6])]

    load_inc_x = min(list(reversed(sorted(abs(load_inc_x))))[0:3])
    load_inc_y = min(list(reversed(sorted(abs(load_inc_y))))[0:3])
    load_inc_z = min(list(reversed(sorted(abs(load_inc_z))))[0:3])

    metric_data['min_torque_inc'][ijk].append(min(load_inc_x, load_inc_y, load_inc_z))


