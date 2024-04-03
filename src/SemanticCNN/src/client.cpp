#include "SemanticCNNClient.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>

using namespace semantic_cnn;
using namespace std;

int main(int argc, char **argv)
{
    // set param
    // google::InitGoogleLogging(argv[0]);
    ros::init(argc, argv, "semantic_client");
    ros::NodeHandle nh("~");

    string bagfile, image_topic;
    nh.getParam("/client_args/bagfile", bagfile);
    nh.getParam("/camera/rgb", image_topic);

    cout << "bagfile_name: " << bagfile << endl;
    cout << "image_topic_name: " << image_topic << endl;

    // connnect to server
    SemanticClient client_node(string("/semantic_server"), true);
    cout << "Connected to semantic action server" << endl;

    // read all images
    vector<cv::Mat> img_vector;
    cv::Mat img;
    sensor_msgs::ImagePtr img_msg_ptr;
    rosbag::Bag bag;
    bag.open(bagfile);
    for (rosbag::MessageInstance const m : rosbag::View(bag))
    {
        if (m.getTopic() == image_topic)
        {
            try
            {
                img_msg_ptr = m.instantiate<sensor_msgs::Image>();
                img = cv_bridge::toCvCopy(img_msg_ptr)->image;
                img_vector.push_back(img);
            }
            catch (cv_bridge::Exception &e)
            {
                ROS_ERROR("Image convert error");
            }
        }
    }
    bag.close();
    cout << "Read images: " << img_vector.size() << endl;

    int batch_size = 10;
    vector<cv::Mat> batch_image;
    for (size_t i = 0; i < img_vector.size(); i++)
    {
        cv::Mat temp = img_vector[i];
        if ((i != 0) && (i % batch_size == 0))
        {
            if (batch_image.size())
            {
                cout << "Request batch id: " << i / batch_size << endl;
                cout << "Size of requst: " << batch_image.size() << endl;
                client_node.RequestSemantic(batch_image);
                batch_image.clear();
            }
        }
        batch_image.push_back(temp);
    }
    ros::spin();
    return 0;
}