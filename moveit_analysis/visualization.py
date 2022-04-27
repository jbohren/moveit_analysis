
class VoxelViz(object):
    """
    Helper class for rendering data as colored voxel grids.
    """

    def __init__(self, topic):
        # Publisher for visualization
        self.marker_pub = rospy.Publisher(topic,MarkerArray,queue_size=10)


    @staticmethod
    def get_cube_list(ns, frame_id, step):
        m = Marker()
        m.id=0
        m.ns=ns
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = 'base_link'
        m.type = Marker.CUBE_LIST
        m.scale.x = step
        m.scale.y = step
        m.scale.z = step
        m.pose.orientation.w=1.0
        m.color.a = 1.0

        return m


    def publish_voxels(
        self,
        ns,
        frame_id,
        voxel_width,
        voxels,
        voxeldata,
        norm=1.0,
        colorfun=(lambda d: matplotlib.cm.viridis(d, 1.0))):
        """
        voxels: dictionary mapping voxel keys to x,y,z positions
        voxeldata: dictionary mapping voxel keys to data
        """

        m = VoxelViz.get_cube_list(ns, frame_id, voxel_width)

        for k in voxels.keys():
            m.points.append(Point(*voxels[k]))
            m.colors.append(ColorRGBA(*colorfun(voxeldata[k]/norm)))

        self.marker_pub.publish(MarkerArray([m]))
