#include "Frame.h"
#include "KeyFrame.h"
#include "Cluster.h"
#include "MapPoint.h"
#include "detect_3d_cuboid/detect_3d_cuboid.h"

using namespace std;
namespace ORB_SLAM3
{
    int Cluster::nId = -1;
    Cluster::Cluster(cv::Rect bbox,KeyFrame* pKF)
    {
        Frame* pFrame = pKF->mFrame;
        nId++;
        mnId = nId;
        mBbox.x = bbox.x;
        mBbox.y = bbox.y;
        mBbox.width = bbox.width;
        mBbox.height = bbox.height;
        mVelocity = cv::Mat::eye(4,4,CV_32FC1);
        mTow = cv::Mat::eye(4,4,CV_32FC1);
        mpFrame=pFrame;
        mnLastAssignedFrameId = pFrame->mnId;
        mCenter = cv::Mat::zeros(3,1,CV_32FC1);
        mpKF = pKF;
    }
    void Cluster::SetPose(cv::Mat pose)
    {
        mTow=pose.clone();
    }
    void Cluster::UpdateBbox(cv::Rect &bbox)
    {
        mBbox = bbox;
    }
    void Cluster::AddMapPoint(MapPoint *pMP)
    {
        unique_lock<mutex> lock(mMutexCluster);
        mspMapPoint.insert(pMP);
    }
    void Cluster::EraseMapPoint(MapPoint *pMP)
    {
        unique_lock<mutex> lock(mMutexCluster);
        mspMapPoint.erase(pMP);
        pMP->SetBadFlag();
    }

    bool Cluster::isStatic()
    {
        return mbStatic;
    }
    void Cluster::updateBbox(cv::Point pt1, cv::Point pt2)
    {
        unique_lock<mutex> lock(mMutexCluster);
        mBbox.x = pt1.x;
        mBbox.y = pt1.y;
        mBbox.width = pt2.x - pt1.x;
        mBbox.height = pt2.y - pt1.y;
    }
    void Cluster::GetSceneOpticalFlow(Frame *pFrame,Frame *pFrameLast)
    {
        unique_lock<mutex> lock(mMutexCluster);
        int N = pFrame->mvObjKeys.size();
        pFrame->vFlow_3d.resize(N);
        // std::vector<Eigen::Vector3d> pts_p3d(N,Eigen::Vector3d(-1,-1,-1)), pts_vel(N,Eigen::Vector3d(-1,-1,-1));
        for (int i = 0; i < N; ++i)
        {
            // get the 3d flow
            int yp=pFrame->mvObjKeys[i].pt.y;
            int xp=pFrame->mvObjKeys[i].pt.x;
            int yc=pFrame->mvObjCorres[i].pt.y;
            int xc=pFrame->mvObjCorres[i].pt.x;
            
            if(pFrameLast->mImDepth.at<float>(yp,xp)<0)
                continue;
            if(pFrame->mImDepth.at<float>(yc,xc)<0)
                continue;
            
            cv::Mat x3D_p = pFrameLast->UnprojectStereo(xp,yp,pFrameLast->mImDepth.at<float>(yp,xp));
            cv::Mat x3D_c = pFrame->UnprojectStereo(xc,yc,pFrame->mImDepth.at<float>(yc,xc));

            // pts_p3d[i] << x3D_p.at<float>(0), x3D_p.at<float>(1), x3D_p.at<float>(2);

            // cout << "3d points: " << x3D_p << " " << x3D_c << endl;

            cv::Point3f flow3d;
            flow3d.x = x3D_c.at<float>(0) - x3D_p.at<float>(0);
            flow3d.y = x3D_c.at<float>(1) - x3D_p.at<float>(1);
            flow3d.z = x3D_c.at<float>(2) - x3D_p.at<float>(2);

            // pts_vel[i] << flow3d.x, flow3d.y, flow3d.z;

            pFrame->vFlow_3d[i] = flow3d;
        }
    }
    bool Cluster::Update3DBbox(cuboid* object_box)
    {
        unique_lock<mutex> lock(mMutexCluster);
        cv::Point3d pt1(0x3f3f3f3f,0x3f3f3f3f,0x3f3f3f3f),pt2(0,0,0);
        if(mBoxCorner.size()==0)
            mBoxCorner.reserve(8);
        for(int i=0;i<8;i++)
        {
            mBoxCorner[i] = cv::Point3f(object_box->box_corners_3d_world(0,i),
                                        object_box->box_corners_3d_world(1,i),
                                        object_box->box_corners_3d_world(2,i));
            pt1.x = min(pt1.x,object_box->box_corners_3d_world(0,i));
            pt1.y = min(pt1.y,object_box->box_corners_3d_world(1,i));
            pt1.z = min(pt1.z,object_box->box_corners_3d_world(2,i));
            pt2.x = max(pt2.x,object_box->box_corners_3d_world(0,i));
            pt2.y = max(pt2.y,object_box->box_corners_3d_world(1,i));
            pt2.z = max(pt2.z,object_box->box_corners_3d_world(2,i));
        }
        mBbox1 = pt1;
        mBbox2 = pt2;
        detected=true;
        return true;
    }
    void Cluster::mergeFrom(Frame* pFrame,Cluster* pCluster)
    {
        unique_lock<mutex> lock(mMutexCluster);
        for(int i=0;i<pFrame->N;i++)
        {
            if(pFrame->mvpCluster[i]==pCluster)
            {
                pFrame->mvpCluster[i] = this;
                pFrame->mvpClusterMapPoint[i]->mpCluster = this;
            }
        }
    }
    void Cluster::mergeFrom(KeyFrame* pFrame,Cluster* pCluster)
    {
        unique_lock<mutex> lock(mMutexCluster);
        for(int i=0;i<pFrame->N;i++)
        {
            if(pFrame->mvpCluster[i]==pCluster)
            {
                pFrame->mvpCluster[i] = this;
                pFrame->mvpClusterMapPoint[i]->mpCluster = this;
            }
        }
    }
    bool Cluster::inBox(cv::Mat pos)
    {
        unique_lock<mutex> lock(mMutexCluster);
        float x = pos.at<float>(0),y = pos.at<float>(1), z = pos.at<float>(2);
        return x>mBbox1.x&&x<mBbox2.x && y>mBbox1.y&&y<mBbox2.y &&z>mBbox1.z&&z<mBbox2.z;
    }
    Cluster::~Cluster()
    {
        nId--;
    }
}