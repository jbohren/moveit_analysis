#!/usr/bin/env python3

"""
load metrics data file
publishes voxels to rviz for a given metric
"""

import matplotlib.cm

import os
import sys
import rclpy
import rclpy.node
import pickle

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from IPython import embed

class VoxelViz(object):
    """
    Helper class for rendering data as colored voxel grids.
    """

    def __init__(self, node, topic):

        self.clock = node.get_clock()

        # Publisher for visualization
        self.marker_pub = node.create_publisher(
                MarkerArray,topic,10)

    def get_cube_list(self, ns, frame_id, width):
        m = Marker()
        m.id=0
        m.ns=ns
        m.header.stamp = self.clock.now().to_msg()
        m.header.frame_id = frame_id
        m.type = Marker.CUBE_LIST
        m.scale.x = width
        m.scale.y = width
        m.scale.z = width
        m.pose.orientation.w=1.0
        m.color.a = 1.0

        return m


    def publish_voxels(
        self,
        analysis,
        ns,
        norm=1.0,
        colorfun=(lambda d: matplotlib.cm.viridis(d, 1.0))):
        """
        voxels: dictionary mapping voxel keys to x,y,z positions
        voxeldata: dictionary mapping voxel keys to data
        """

        frame_id = analysis.base_link


        m = self.get_cube_list(ns, frame_id, analysis.step)

        # iterate over pos/orientation
        for ipos in range(analysis.metrics.shape[0]):
            for iori in range(analysis.metrics.shape[1]):
                sample = analysis.metrics[ipos,iori]

                if sample is None:
                    continue

                metric = getattr(sample, ns)
                print(metric)
                color = colorfun(metric/norm)

                m.points.append(Point(
                            x=sample.F[0,3],
                            y=sample.F[1,3],
                            z=sample.F[2,3]))
                m.colors.append(ColorRGBA(
                    r=color[0],
                    g=color[1],
                    b=color[2],
                    a=0.2))

        self.marker_pub.publish(MarkerArray(markers=[m]))

def main():

    analysis_path = sys.argv[1]

    if not os.path.exists(analysis_path):
        exit(1)

    with open(analysis_path, 'rb') as analysis_file:
        analysis = pickle.load(analysis_file)

    # embed()

    rclpy.init()

    node = rclpy.node.Node('view_metrics')
    viz = VoxelViz(node, 'chain_metrics')

    viz.publish_voxels(analysis, 'manipulability', norm=0.006)


    rclpy.spin(node)

if __name__ == '__main__':
    main()
