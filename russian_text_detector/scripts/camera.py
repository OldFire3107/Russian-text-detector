#!/usr/bin/env python

import numpy as np
import rospy
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def camera():
    pub = rospy.Publisher('videofeed', Image, queue_size=1)
    bridge = CvBridge()
    rospy.init_node('camera', anonymous=True)

    video = cv2.VideoCapture(0)
    fps = video.get(cv2.CAP_PROP_FPS)

    #rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        _, frame = video.read()

        pub.publish(bridge.cv2_to_imgmsg(frame))
        rospy.loginfo('sent image')
        #rate.sleep()

        cv2.imshow('vid', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    try:
        camera()
    except rospy.ROSInterruptException:
        pass
