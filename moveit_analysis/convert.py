
from typing import overload,Tuple

import numpy as np
import PyKDL as kdl

from geometry_msgs.msg import Pose, Transform

"""
Conversion methods for geometric types.

Example:

>>> import PyKDL
>>> import moveit_analysis.conversions as convert
>>> frame = PyKDL.Frame()
>>> pose_msg = convert.KDL.Frame.ToPoseMsg(frame)

"""

def kdl_frame_from(
        msg:    Pose = None,
        p_xyz:  Tuple[float,float,float] = None,
        q_wxyz: Tuple[float,float,float,float] = None
        ):

    if msg:
        return kdl.Frame(
                kdl.Vector(
                    msg.position.x,
                    msg.position.y,
                    msg.position.z),
                kdl.Rotation.Quaternion(
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w))

    elif p_xyz and q_wxyz:
        return kdl.Frame(
                kdl.Rotation.Quaternion(
                    q_wxyz[1],
                    q_wxyz[2],
                    q_wxyz[3],
                    q_wxyz[0]),
                kdl.Vector(*p_xyz))

    raise NotImplementedError()

def to_joint_mat(msg: Pose):
    return kdl.Frame(
            kdl.Vector(
                msg.position.x,
                msg.position.y,
                msg.position.z),
            kdl.Rotation.Quaternion(
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w))

def to_pose_msg(f: kdl.Frame):
    """
    Return a ROS Pose message for the KDL Frame f.
    """
    p = Pose()
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = f.M.GetQuaternion()
    p.position.x = f.p[0]
    p.position.y = f.p[1]
    p.position.z = f.p[2]
    return p

def to_transform_msg(f: kdl.Frame):
    """
    Return a ROS Transform message for the KDL Frame f.
    """
    p = Transform()
    p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w = f.M.GetQuaternion()
    p.translation.x = f.p[0]
    p.translation.y = f.p[1]
    p.translation.z = f.p[2]
    return p

@overload
def to_frame_mat(F: kdl.Frame):
    return np.matrix([
        [_ for _ in F.M.UnitX()]+[0.0],
        [_ for _ in F.M.UnitY()]+[0.0],
        [_ for _ in F.M.UnitZ()]+[0.0],
        [F.p.x(), F.p.y(), F.p.z(), 1.0]]).T

class PoseMsg:
    class To:
        class KDL:
            def Frame(msg):
                return kdl.Frame(
                        kdl.Rotation.Quaternion(
                            msg.orientation.x,
                            msg.orientation.y,
                            msg.orientation.z,
                            msg.orientation.w),
                        kdl.Vector(
                            msg.position.x,
                            msg.position.y,
                            msg.position.z)
                        )


class KDL:

    class JntArray:
        class To:
            @staticmethod
            def Mat(q):
                return np.array([_ for _ in q], dtype=np.double)

    class Frame:
        class To:
            @staticmethod
            def PoseMsg(f):
                """
                Return a ROS Pose message for the KDL Frame f.
                """
                p = Pose()
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = f.M.GetQuaternion()
                p.position.x = f.p[0]
                p.position.y = f.p[1]
                p.position.z = f.p[2]
                return p

            @staticmethod
            def TransformMsg(f):
                """
                Return a ROS Transform message for the KDL Frame f.
                """
                p = Transform()
                p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w = f.M.GetQuaternion()
                p.translation.x = f.p[0]
                p.translation.y = f.p[1]
                p.translation.z = f.p[2]
                return p

            @staticmethod
            def Mat(F):
                return np.matrix([
                    [_ for _ in F.M.UnitX()]+[0.0],
                    [_ for _ in F.M.UnitY()]+[0.0],
                    [_ for _ in F.M.UnitZ()]+[0.0],
                    [F.p.x(), F.p.y(), F.p.z(), 1.0]]).T

    class Rotation:
        class To:
            @staticmethod
            def Mat(R):
                """
                Return a numpy matrix representation of a KDL Rotation.
                """
                return np.matrix([
                    [_ for _ in R.UnitX()],
                    [_ for _ in R.UnitY()],
                    [_ for _ in R.UnitZ()]]).T

    class Jacobian:
        class To:
            @staticmethod
            def Mat(J):
                return np.matrix(
                        [[J[i,j] for j in range(J.columns()) ]
                        for i in range(6)])

class Array:
    class To:
        class KDL:
            @staticmethod
            def JntArray(a):
                """Convert a numpy array to a KDL JntArray"""
                ja = kdl.JntArray(len(a))
                for i,v in enumerate(a):
                    ja[i] = v
                return ja


