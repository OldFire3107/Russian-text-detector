#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from custom_msg.msg import Result

FindString = 'ОФИГЕННО!'
pub = rospy.Publisher('final_result', Result, queue_size=10)

def callback(data):
    text = data.isSame.decode('utf-8')
    # rospy.loginfo(rospy.get_caller_id() + "Word is %s", text)
    print(text)
    msg = Result()

    if FindString == data.isSame:
        msg.isSame = 'Same'
        msg.x = data.x
        msg.y = data.y
        pub.publish(msg)
    else:
        msg.isSame = 'Not Same'
        msg.x = data.x
        msg.y = data.y
        pub.publish(msg)

def matching():
    rospy.init_node('matching', anonymous=True)
    rospy.Subscriber('detection_result', Result, callback)

    rospy.spin()


if __name__ == '__main__':
    try:
        matching()
    except rospy.ROSInterruptException:
        pass
