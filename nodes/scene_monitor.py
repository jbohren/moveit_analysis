#!/usr/bin/env python

import rospy

from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene


def main():
    rospy.init_node('planning_scene_monitor')

    get_scene = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)

    r = rospy.Rate(10)
    
    while not rospy.is_shutdown():

        # Request planning scene
        scene = get_scene(PlanningSceneComponents(
                PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS+
                PlanningSceneComponents.WORLD_OBJECT_NAMES)).scene

        objects = [co.id for co in scene.world.collision_objects]
        attached_objects = {aco.object.id: aco.link_name for aco in scene.robot_state.attached_collision_objects}

        print('Objects:')
        for co in objects:
            print('{}{}'.format(co, ' (attached to {})'.format(attached_objects[co]) if co in attached_objects else ''))
        for co, link_name in attached_objects.items():
            print('{} (attached to {})'.format(co, link_name))

        r.sleep()

if __name__ == '__main__':
    main()
