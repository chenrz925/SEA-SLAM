/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Imu.h>

#include <opencv2/core/core.hpp>

#include "System.h"
#include "ImuTypes.h"
#include "Common.h"
#include "Config.h"
#include "Semantic.h"
using namespace std;
class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM):mpSLAM(pSLAM){}

    void GrabImage(const sensor_msgs::ImageConstPtr& msg);

    ORB_SLAM3::System* mpSLAM;
};

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    mpSLAM->TrackMonocular(cv_ptr->image,cv_ptr->header.stamp.toSec());
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mono");
    ros::start();
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    bool bEqual = false;
    if (argc < 3|| argc > 4)
    {
        cerr << endl
             << "Usage: rosrun ORB_SLAM3 Mono path_to_vocabulary path_to_settings  saving_path" << endl;
        ros::shutdown();
        return 1;
    }

    if (argc == 4)
    {
        ORB_SLAM3::Config::GetInstance()->IsSaveResult(true);
        ORB_SLAM3::Config::GetInstance()->createSavePath(string(argv[3]));
    }
    else{
        ORB_SLAM3::Config::GetInstance()->IsSaveResult(false);
    }
    // thread(spin_thread);

    cout << "===========================" << endl;
    cout << "argv[1]: " << argv[1] << endl;
    cout << "argv[2]: " << argv[2] << endl;
    cout << "argv[3]: " << argv[3] << endl;
    cout << "===========================" << endl;
    // Create SLAM system. It initializes all system threads and gets ready to process frames.

    float fInitDelay, fFrameDelay;
    int iInitFrames, iImageDelay;
    int bSaveImgResult,bUseDensemapping,bUseViewer;
    string strSemanticConfigPath;
    nh.getParam("frame_delay", fFrameDelay);
    nh.getParam("init_delay", fInitDelay);
    nh.getParam("init_frame", iInitFrames);
    nh.getParam("image_delay", iImageDelay);
    nh.getParam("semantic_config_path", strSemanticConfigPath);
    nh.getParam("save_image_result", bSaveImgResult);
    // nh.getParam("use_densemapping", bUseDensemapping);
    nh.getParam("use_viewer", bUseViewer);

    ORB_SLAM3::Semantic::mstrConfigPath = strSemanticConfigPath;
    ORB_SLAM3::Semantic::mbSaveResult = bSaveImgResult;
    ORB_SLAM3::Semantic::mbViwer = bUseViewer;
    ORB_SLAM3::Viewer::IMAGE_DELAY = iImageDelay;
    // Create SLAM system. It initializes all system threads and gets ready to process frames.

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, bUseViewer);
    SLAM.mstrSavingPath = string(argv[4]);
    cout<<"system init finished"<<endl;

    ORB_SLAM3::Semantic::GetInstance()->Run();
    cout<<"semantic part init finished"<<endl;

    ImageGrabber igb(&SLAM);

    // Maximum delay, 5 seconds
    // ros::Subscriber sub_imu = nh.subscribe("/imu", 1000, &ImuGrabber::GrabImu, &imugb);
    ros::Subscriber sub_img0 = nh.subscribe("/image", 1, &ImageGrabber::GrabImage, &igb);

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}