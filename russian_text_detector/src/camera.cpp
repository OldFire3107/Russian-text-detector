#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    VideoCapture vid(0);
    ros::init(argc, argv, "camera");
    Mat frame;

    ros::NodeHandle camera;
    image_transport::ImageTransport it_(camera);
    image_transport::Publisher pub = it_.advertise("videofeed", 1);
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    cv_ptr->encoding = "bgr8";
    cv_ptr->header.frame_id = "videofeed";

    while (ros::ok())
    {
        vid >> frame;
        ros::Time time = ros::Time::now();
        cv_ptr->header.stamp = time;
        cv_ptr->image = frame;

        pub.publish(cv_ptr->toImageMsg());

        ros::spinOnce();;
    }

    return 0;
}
