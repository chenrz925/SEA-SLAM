#include "Atlas.h"
#include "ORBmatcher.h"
#include "Common.h"
#include "Cluster.h"
#include "Converter.h"
#include <munkres.h>
// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

#include <future>
#include "line_lbd/line_lbd_allclass.h"
#include "detect_3d_cuboid/detect_3d_cuboid.h"

using namespace std;
namespace ORB_SLAM3
{
    // line_lbd_detect：线检测类，定义一个线检测类的对象 line_lbd_ptr.
    // line_lbd_detect* Atlas::line_lbd_ptr = new line_lbd_detect(1,2.0); 

    // detect_3d_cuboid* Atlas::detect_cuboid_obj = new detect_3d_cuboid(721.5377,721.5377,609.5593,172.854);

    void Atlas::UpdateClusters(KeyFrame *pKF, Frame *pFrame, cv::Mat mask,vector<ObjectSet> all_object_cuboids)
    {
        if (pKF->mObjectNum == 0)
            return;
        
        std::vector<Cluster *> mvpCluster = GetCurrentMap()->GetAllClusters();
        if (mvpCluster.size() == 0)
        {

            ORBmatcher objmatcher(0.9, true);
            for (int i = 0; i < pKF->mBbox.size(); i++)
            {
                ObjectSet temp_box= all_object_cuboids[i];
                Cluster *pCluster = new Cluster(pKF->mBbox[i], pKF);
                if(temp_box.size()!=0) 
                    pCluster->Update3DBbox(temp_box[0]);
                GetCurrentMap()->AddCluster(pCluster);
                mClusterMapid[i] = pCluster;
            }

            for (int i = 0; i < pKF->N; i++)
            {
                if (pKF->mvDepth[i] > 0) //||pKF->mvDepth[i]>40)
                {
                    int objectid = mask.at<uchar>(pKF->mvKeys[i].pt.y, pKF->mvKeys[i].pt.x);
                    if (objectid != 0)
                    {
                        cv::Mat x3D = pKF->UnprojectStereo(i);
                        if (x3D.empty())
                            continue;
                        Cluster *pCluster = GetCurrentMap()->GetCluster(objectid - 1);
                        if (!pCluster)
                            continue;
                        if(pCluster->mBoxCorner.size()==0)
                        {
                            MapPoint *pMP = new MapPoint(x3D,pKF,GetCurrentMap());
                            pMP->AddObservation(pKF, i);
                            pMP->ComputeDistinctiveDescriptors();
                            pMP->UpdateNormalAndDepth();
                            pCluster->AddMapPoint(pMP);
                            pMP->mpCluster = pCluster;
                            pKF->AddCluster(i, pCluster);
                            pKF->AddClusterMapPoint(i, pMP);
                        }
                        else
                        {
                            if (pCluster->inBox(x3D))
                            {
                                MapPoint *pMP = new MapPoint(x3D,pKF,GetCurrentMap());
                                pMP->AddObservation(pKF, i);
                                pMP->ComputeDistinctiveDescriptors();
                                pMP->UpdateNormalAndDepth();
                                pCluster->AddMapPoint(pMP);
                                pMP->mpCluster = pCluster;
                                pKF->AddCluster(i, pCluster);
                                pKF->AddClusterMapPoint(i, pMP);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            if (pKF->mnId < mpKFSemantic->mnId)
                return;
            AssociateNewBBox(pKF, mask, all_object_cuboids);
        }
        /*
        else
        {
            int n=mvpCluster.size();
            cv::Mat flowObject = pKF->mFrame->mImMask;
            cv::Mat imMask=pKF->mImMask.clone();
            // cv::Mat flow,flowObject;
            cv::imwrite("/out/debug/semanticmask.png",imMask>0);
            // cv::calcOpticalFlowFarneback(mpKFSemantic->mImGray,pKF->mImGray,flow,0.5, 3, 15, 3, 5, 1.2, 0);
            if(flowObject.empty())
                return ;
            // flowObject=cv::Mat::zeros(flow.size(),CV_8UC1);
            // UpdateMask(mpKFSemantic->mImMask,flow,flowObject);
            // cv::erode(pKF->mImMask,imMask,10);
            cv::imwrite("/out/debug/flowmask.png",flowObject>0);

            vector<int> clusterMap(n,-1);
            vector<int> clusterIoU(n,0);
            for(int i=0;i<n;i++)
            {
                float maxx_iou = 0.5;
                for(int j=0;j<pKF->mBbox.size();j++)
                {
                    // if(clusterMap[j]!=-1)
                    //     continue;
                    cv::Mat matflow=(flowObject==(mvpCluster[i]->mnId));
                    cv::Mat matmask=(imMask==(j+1));
                    float roi = cv::countNonZero(matflow&matmask);
                    float numsemantic = cv::countNonZero(matmask|matflow);
                    float numall = cv::countNonZero(matmask|matflow);
                    if(roi/numall>maxx_iou){
                        maxx_iou=roi/numall;
                        if(maxx_iou>clusterIoU[i]){
                            clusterMap[i]=j+1;
                            clusterIoU[i]=maxx_iou;
                        }
                    }
                }
            }
            for(int i=0;i<n;i++)
            {
                for(int j=i+1;j<n;j++)
                {
                    if(clusterMap[i]==clusterMap[j])
                    {
                        if(clusterIoU[i]<clusterIoU[j])
                        {
                            clusterMap[i]=-1;
                        }
                        else
                        {
                            clusterMap[j]=-1;
                        }
                    }
                }
            }
            cv::Mat newMask = cv::Mat::zeros(pKF->mImMask.size(),CV_8UC1);
            for(int i=0;i<n;i++)
            {
                if(clusterMap[i]==-1)
                {
                    cv::Mat updatemask = (flowObject==mvpCluster[i]->mnId);
                    flowObject.copyTo(newMask,updatemask);
                }
                else{
                    cv::Mat updatemask = (imMask==clusterMap[i]);
                    cv::Mat copyMask = cv::Mat::ones(pKF->mImMask.size(),CV_8UC1)*(mvpCluster[i]->mnId);
                    copyMask.copyTo(newMask,updatemask);
                    copyMask.copyTo(mvpCluster[i]->mImMask,updatemask);
                }
            }
            for(int i=0;i<pKF->mObjectNum;i++)
            {
                vector<int>::iterator result = find(clusterMap.begin( ),clusterMap.end( ), i+1 );
                if(result==clusterMap.end())
                {
                    n++;
                    Cluster* pCluster = new Cluster(pKF->mBbox[i],pKF->mFrame);
                    cv::Mat updatemask = (imMask==(i+1));
                    cv::Mat copyMask = cv::Mat::ones(newMask.size(),CV_8UC1)*(pCluster->mnId);
                    cv::dilate(copyMask, copyMask, 15);
                    copyMask.copyTo(newMask,updatemask);
                    pCluster->mpKF=pKF;
                    mvpCluster.push_back(pCluster);
                }
            }
            pKF->mImMask=newMask;

        }
        */
        mpKFSemantic = pKF;
    }
    void Atlas::AssociateNewBBox(KeyFrame *pKF, cv::Mat mask, std::vector<std::vector<cuboid *>> all_object_cuboids)
    {
        mClusterMapid.clear();
        std::vector<Cluster *> mvpCluster = GetCurrentMap()->GetAllClusters();
        int n = mvpCluster.size();
        munkres::Matrix<int> munkresInput;
        munkres::Matrix<int> munkresData;
        munkresData.resize(pKF->mObjectNum, n);
        munkresData.clear();
        for (int i = 0; i < pKF->N; i++)
        {
            Cluster *pCluster = pKF->mvpCluster[i];
            if (pCluster)
            {
                int cid = mask.at<uchar>(pKF->mvKeys[i].pt.y, pKF->mvKeys[i].pt.x) - 1;
                if (cid != -1)
                {
                    int poid = pCluster->mnId;
                    munkresData(cid, poid) += 1;
                }
            }
        }
        std::vector<bool> rowValid(munkresData.rows(), false);
        for (int r = 0; r < munkresData.rows(); ++r)
        {
            for (int c = 0; c < munkresData.columns(); ++c)
            {
                if (munkresData(r, c) != 0)
                {
                    rowValid[r] = true;
                    break;
                }
            }
        }
        munkresInput = munkresData;
        munkresData.revert();
        Munkres<int> munkresSolver;
        munkresSolver.solve(munkresData);
        for (int r = 0; r < munkresData.rows(); ++r)
        {
            if (rowValid[r])
                continue;
            for (int c = 0; c < munkresData.columns(); ++c)
            {
                munkresData(r, c) = -1;
            }
        }
        std::map<int, Cluster *> clusterAssociation;
        std::set<Cluster *> spNewClusters;
        std::map<Cluster *, int> mNewClusterid;
        for (int cid = 0; cid < pKF->mObjectNum; cid++)
        {
            bool matched = false;
            for (auto cluster : mvpCluster)
            {
                int poid = cluster->mnId;
                if (munkresData(cid, poid) == 0 && munkresInput(cid, poid) > 0)
                {
                    clusterAssociation[cid] = cluster;
                    cluster->mBbox = pKF->mBbox[cid];
                    mClusterMapid[cid] = cluster;
                    cluster->Update3DBbox(all_object_cuboids[cid][0]);
                    matched = true;
                    break;
                }
            }
            if (!matched)
            {
                auto *newCluster = new Cluster(pKF->mBbox[cid], pKF);
                clusterAssociation[cid] = newCluster;
                spNewClusters.insert(newCluster);
                mClusterMapid[cid] = newCluster;
                mNewClusterid[newCluster] = cid;
            }
        }
        for (int i = 0; i < clusterAssociation.size(); i++)
        {
            clusterAssociation[i]->Update3DBbox(all_object_cuboids[i][0]);
        }
        for (int i = 0; i < pKF->N; i++)
        {
            if (pKF->mvpCluster[i])
                continue;
            if (pKF->mvDepth[i] < 0)
                continue;
            int cid = mask.at<uchar>(pKF->mvKeys[i].pt.y, pKF->mvKeys[i].pt.x) - 1;
            if (cid == -1)
                continue;
            cv::Mat x3D = pKF->UnprojectStereo(i);
            if (x3D.empty())
                continue;
            if (clusterAssociation[cid]->inBox(x3D))
            {
                MapPoint *pMP = new MapPoint(x3D,pKF,GetCurrentMap());
                pMP->AddObservation(pKF, i);
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
                pMP->mpCluster = clusterAssociation[cid];
                clusterAssociation[cid]->AddMapPoint(pMP);
                pKF->AddCluster(i, clusterAssociation[cid]);
                pKF->AddClusterMapPoint(i, pMP);
            }
        }
        std::map<Cluster *, Cluster *> clusterMerging;
        for (auto *cc : spNewClusters)
        {
            bool isBad = false;
            for (auto *oldp : mvpCluster)
            {
                int nOverlaps = 0;
                int nTotal = 0;
                for (auto poit = cc->mspMapPoint.begin(); poit != cc->mspMapPoint.end(); ++poit)
                {
                    if (*poit == nullptr)
                        continue;
                    nTotal++;
                    if (oldp->inBox((*poit)->GetWorldPos()))
                    {
                        nOverlaps++;
                    }
                }
                float overlapRatio = (float)nOverlaps / nTotal;
                if (overlapRatio > 0.6)
                {
                    clusterMerging[cc] = oldp;
                    oldp->mBbox = pKF->mBbox[mNewClusterid[cc]];
                    isBad = true;
                    break;
                }
            }
            if (isBad)
            {
                clusterMerging[cc]->mergeFrom(pKF,cc);
                if (mNewClusterid.count(cc))
                {
                    mClusterMapid[mNewClusterid[cc]] = clusterMerging[cc];
                }
            }
        }
        for (auto *cc : spNewClusters)
        {
            if (clusterMerging.count(cc) != 0)
            {
                delete cc;
            }
            else
            {
                cc->mnId = (n++);
                GetCurrentMap()->AddCluster(cc);
            }
        }
    }
}
/*

 bool istereo = mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR;
    future<int> trackObject = async(launch::async, [this,th,istereo]{
        int nObjectMatches=0;
        ORBmatcher objmatcher(0.9, true);

        for(auto cluster:mpAtlas->mvpCluster)
        {
            cluster->mvpMapPointLast.clear();
            cluster->mvpMapPointLast.resize(cluster->mvpMapPoint.size());
            cluster->mvpMapPointLast = cluster->mvpMapPoint;
            cluster->mvpMapPoint.clear();
            if(cluster->mnLastAssignedFrameId==mLastFrame.mnId){
                nObjectMatches += objmatcher.SearchObjectByProjection(mCurrentFrame, mLastFrame,cluster, 2 * th, istereo);
                Optimizer::ObjectOptimization(&mCurrentFrame,cluster);
                cluster->mnLastAssignedFrameId=mCurrentFrame.mnId;
            }
        }
        Config::GetInstance()->saveImage(mCurrentFrame.mImMask,"mask",to_string(mCurrentFrame.mnId)+".png");
    
        if(mCurrentFrame.mImMask.rows!=0){
            for(int i=0;i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvDepth[i]<0) continue;
                cv::Point kp = mCurrentFrame.mvKeys[i].pt;
                int objectid = mCurrentFrame.mImMask.at<uchar>(kp.y,kp.x);
                if(objectid!=0&&mCurrentFrame.mvpCluster[i]==nullptr)
                {
                    mCurrentFrame.mvpCluster[i] = mpAtlas->mClusterMapid[objectid-1];
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    MapPoint *newMP;
                    if(pMP)
                    {
                        newMP = new MapPoint(pMP->GetWorldPos(),mpAtlas->GetCurrentMap(),&mCurrentFrame,i);
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
                    else
                    {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                        newMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mCurrentFrame,i);
                    }
                    mCurrentFrame.mvpCluster[i]->AddMapPoint(i,newMP);
                }
            }
        }
        return nObjectMatches;
    });
    */