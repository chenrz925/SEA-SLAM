#ifndef CLUSTER_H
#define CLUSTER_H
#include "Common.h"
#include <mutex>
using namespace std;
class cuboid;
namespace ORB_SLAM3
{
    class MapPoint;
    class KeyFrame;
    class Frame;
    class Cluster{
        public:
            int mnId;
            static int nId;
            set<MapPoint*> mspMapPoint;
            cv::Mat mTow;
            cv::Mat mImMask;
            cv::Rect mBbox;
            cv::Mat mCenter;
            cv::Point3d mBbox1 = cv::Point3d(0.,0.,0.);
            cv::Point3d mBbox2 = cv::Point3d(0.,0.,0.);
            vector<cv::Point3d> mBoxCorner;
            mutex mMutexCluster;
            bool mbStatic=true;
            bool detected = false;
            KeyFrame* mpKF;
            Frame* mpFrame;
            cv::Mat mVelocity;
            int mnLastAssignedFrameId;
            map<int,int> matchid;

            Cluster();
            Cluster(cv::Rect bbox,KeyFrame* pFrame);
            ~Cluster();
            void EraseMapPoint(MapPoint* pMP);
            void UpdateBbox(cv::Rect &bbox);
            void AddMapPoint(MapPoint* pMP);
            bool isStatic();
            void updateBbox(cv::Point pt1,cv::Point pt2);
            void GetSceneOpticalFlow(Frame *pFrame,Frame *pFrameLast);
            void SetPose(cv::Mat pose);
            bool Update3DBbox(cuboid* object_box);
            void mergeFrom(Frame* pFrame,Cluster* pCluster);
            void mergeFrom(KeyFrame* pFrame,Cluster* pCluster);

            bool inBox(cv::Mat pos);

           
    };
}

#endif