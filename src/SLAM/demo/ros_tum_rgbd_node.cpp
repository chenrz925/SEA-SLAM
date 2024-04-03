#define DEBUG 0
#include "Common.h"
#include "Config.h"
#include "System.h"
#include "Semantic.h"
#include "ImgProcessor.h"
using namespace std;
using namespace ORB_SLAM3;
void spin_thread()
{
    ros::spinOnce();
    usleep(1);
}

int main(int argc, char **argv)
{
#ifdef DEBUG
    printf("===============using debug print================");
#endif
    google::InitGoogleLogging(argv[0]);
    ros::init(argc, argv, "tum");
    ros::start();
    ros::NodeHandle nh("~");

    // check args
    if (argc < 5)
    {
        cerr << endl
             << "Usage: EXE ./Dynamic_RGBD path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        ros::shutdown();
        return 1;
    }
    cout << "===========================" << endl;
    cout << "argv[1]: " << argv[1] << endl;
    cout << "argv[2]: " << argv[2] << endl;
    cout << "argv[3]: " << argv[3] << endl;
    cout << "argv[4]: " << argv[4] << endl;
    cout << "argv[5] result path: " << argv[5] << endl;
    cout << "===========================" << endl;

    if (argc == 6)
    {
        
        Config::GetInstance()->IsSaveResult(true);
        Config::GetInstance()->createSavePath(string(argv[5]));
    }
    else
    {
        Config::GetInstance()->IsSaveResult(false);
    }
    thread(spin_thread);

    vector<string> vstrImageFileRGB;
    vector<string> vstrImageFileD;
    vector<double> vTimestamps;
    string strAssociationFile = string(argv[4]);

    Config::GetInstance()->LoadTUMDataset(strAssociationFile, vstrImageFileRGB, vstrImageFileD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFileRGB.size();
    if (vstrImageFileRGB.empty())
    {
        cerr << endl
             << "No images found in provided path." << endl;
        return 1;
    }
    else if (vstrImageFileD.size() != vstrImageFileD.size())
    {
        cerr << endl
             << "Different number of images for rgb and depth." << endl;
        return 1;
    }

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
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, bUseViewer);
    cout<<"system init finished"<<endl;

    Semantic::GetInstance()->Run();
    cout<<"semantic part init finished"<<endl;

    float avg_time_perframe = 0;
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);
    cv::Mat imRGB, imD,imDU;

    // lime* imgPreProcessor;
    // imgPreProcessor = new lime();
    ros::Duration(5).sleep();
    for (int i = 0; i < nImages; i++)
    {
        imRGB = cv::imread(string(argv[3]) + "/" + vstrImageFileRGB[i], cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3]) + "/" + vstrImageFileD[i], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[i];
        // imRGB = imgPreProcessor->lime_enhance(imRGB);
        if (imRGB.empty())
        {
            cerr << endl
                 << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFileRGB[i] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
#else
        chrono::monotonic_clock::time_point t1 = chrono::monotonic_clock::now();
#endif
        // cout << "===============Tracking Frame " << i << "============" << endl;
        // cout << "imRGB size: " << imRGB.size() << endl;
        // cout << "imD size: " << imD.size() << endl;
        // cout << "tframe: " << tframe << endl;
        SLAM.TrackRGBD(imRGB, imD, tframe);
        // cout << "===============Frame " << i << " Finished============" << endl;

#ifdef COMPILEDWITHC11
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
#else
        chrono::monotonic_clock::time_point t2 = chrono::monotonic_clock::now();
#endif
        double ttrack = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();

        vTimesTrack[i] = ttrack;
        avg_time_perframe += ttrack;

        // Manually add delay to evaluate TUM, because TUM dataset is very short
        if (i < iInitFrames)
        {
            usleep(fInitDelay);
        }
        else
        {
            usleep(fFrameDelay);
        }
        // Wait to load the next frame
        double T = 0;
        if (i < nImages - 1)
            T = vTimestamps[i + 1] - tframe;
        else if (i > 0)
            T = tframe - vTimestamps[i - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);
    }
    avg_time_perframe /= vTimesTrack.size();
    cout << "Tracking per fram : " << avg_time_perframe << "s" << endl;
    cout << "===============Tracking Finished============" << endl;

    cout << "===============Final Stage============" << endl;
    // Stop semantic thread
    Semantic::GetInstance()->RequestFinish();

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int i = 0; i < nImages; i++)
    {
        totaltime += vTimesTrack[i];
    }
    cout << "-------" << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    LOG(INFO) << "median tracking time: " << vTimesTrack[nImages / 2] * 1000 << "ms";
    cout << "mean tracking time: " << totaltime / nImages << endl;
    LOG(INFO) << "mean tracking time: " << totaltime / nImages * 1000 << "ms";

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM(string(argv[5])+"/CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM(string(argv[5])+"/KeyFrameTrajectory.txt");

    ros::shutdown();
    return 0;
}
