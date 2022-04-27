
from typing import Sequence,Tuple

import numpy as np

from math import pi,sin,cos

import PyKDL as kdl

from moveit_analysis.stateless import PlannerInterface
from moveit_analysis.stateful import WorldInterface

def regular_spherical_point_distribution(r, N):
    """
    Generate a list of N points arrange uniformly over a sphere of radius r.

    r: radius
    N: number of points
    """
    a = 4*pi*r*r / N
    d = sqrt(a)
    Mth = int(round(pi/d))
    dth = pi/Mth
    dph = a/dth
    for m in range(Mth):
        th = pi*(m + 0.5) / Mth
        Mph = int(round(2*pi*sin(th)/dph))
        for n in range(Mph):
            ph = 2*pi*n/Mph
            yield (
                r*np.array([
                    sin(th)*cos(ph),
                    sin(th)*sin(ph),
                    cos(th)]))

def regular_spherical_orientation_distribution(n_tip_vectors, n_tip_rolls):
    """
    Generate a near-uniformly sampled set of orientations
    Number of outputs is quadratic in the number of argumnets
    """
    rotx = [
        kdl.Vector(*rx) for rx
        in WorkspaceSampler.regular_spherical_point_distribution(1.0, n_tip_vectors)]
    roty = [kdl.Vector(0,0,1)*v for v in rotx]
    roty = [v/v.Norm() for v in roty]
    rotz = [x*y for x,y in zip(rotx,roty)]
    tip_orientations = [
        kdl.Rotation(rx,ry,rz)
        for rx,ry,rz
        in zip(rotx,roty,rotz)]
    tip_rolls = [
        kdl.Rotation.RotX(a)
        for a in range(n_tip_rolls)]

    for tip_orientation in list(tip_orientations):
        for tip_roll in list(tip_rolls):
            yield tip_orientation * tip_roll


def cartesian_positions(
        x_range: Tuple[float,float],
        y_range: Tuple[float,float],
        z_range: Tuple[float,float],
        step: float
):
    """
    Generate a list of points covering a cartesian workspace
    Number of outputs is quartic in the number of arguments

    x_positions: tuple (min, max) x position
    y_positions: tuple (min, max) x position
    z_positions: tuple (min, max) x position

    returns a tuple x,y,z
    """

    # Compute set of orientations

    return [(x,y,z)
            for x in np.arange(*x_range, step)
            for y in np.arange(*y_range, step)
            for z in np.arange(*z_range, step)]

class WorkspaceAnalyzer:
    """
    Computes postures over a given cartesian sample space.
    """

    def __init__(self, node):

        # Store handle to node
        self.node = node

        # Interface helpers
        self.planner_interface = PlannerInterface(self.node)
        self.world_interface = WorldInterface(self.node)

    def compute_postures(
            self,
            chain_model,
            positions,
            orientations
            ):

        self.node.get_logger().info(f'Computing postures for chain: {chain_model.base_link} --> {chain_model.tip_link}')

        self.node.get_logger().info(str(chain_model.joint_names))

        # Pre-allocate postures array
        postures = np.ndarray((len(positions), len(orientations)), dtype=object)

        # Initialize seed position for ik solver
        # TODO: pick seed with high manipulability?
        seed_state = chain_model.get_robot_state([0.1]*len(chain_model.joint_names))

        # Iterate over sample space by position/orientation
        # statusbar = tqdm(enumerate(positions), total=len(positions))
        for ipos, (x,y,z) in enumerate(positions):
            for iori, (qx, qy, qz, qw) in enumerate(orientations):

                # TODO: do we need to set the joint state?
                self.world_interface.set_joint_state(seed_state)

                pose = kdl.Frame(
                        kdl.Rotation.Quaternion(x=qx,y=qy,z=qz,w=qw),
                        kdl.Vector(x,y,z))

                ik_solution_state = self.planner_interface.get_ik(
                    seed_state,
                    chain_model.planning_group,
                    chain_model.tip_link,
                    frame_id         = chain_model.base_link,
                    pose             = pose,
                    avoid_collisions = True)

                # Skip if no solution found
                # TODO: try other seed positions if IK fails
                if ik_solution_state is None:
                    continue

                # Update seed
                seed_state.joint_state = ik_solution_state.joint_state

                # Save posture solution as np array
                postures[ipos, iori] = np.array(chain_model.get_posture(ik_solution_state))

        return postures

