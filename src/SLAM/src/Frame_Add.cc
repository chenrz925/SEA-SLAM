#include "Frame.h"
#include "MapPoint.h"
#include "Cluster.h"

using namespace std;
namespace ORB_SLAM3
{
    void Frame::AddClusterMapPoint(int i,MapPoint* pMP)
    {
        // unique_lock<std::mutex> lock(mMutexCluster);
        mvpClusterMapPoint[i]=pMP;
    }
    void Frame::AddCluster(int i,Cluster* pCluster)
    {
        // unique_lock<std::mutex> lock(mMutexCluster);
        mvpCluster[i]=pCluster;
    }
    bool Frame::checkNeighbor(int i)
    {
        if(mImMask.empty())
            return true;
        int dx[10] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
        int dy[10] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
        cv::Point kp = mvKeys[i].pt;
        // for (int j = 0; j < 9; j++){
        for(int j=-5;j<=5;j++)
        {
            for(int i=-5;i<=5;i++)
            {
                if(kp.y+j<0||kp.y+j>mImMask.rows) continue;
                if(kp.x+i<0||kp.x+i>mImMask.cols) continue;
                if(mImMask.at<uchar>(kp.y + j, kp.x + i)!=0)
                    return false;
            }
        }    
        return true;
    }
    bool Frame::checkMask(MapPoint* pMP)
    {
        if(mImMask.empty())
            return true;
        cv::Mat P = pMP->GetWorldPos();
        // 3D in camera coordinates
        const cv::Mat Pc = mRcw * P + mtcw;
        const float Pc_dist = cv::norm(Pc);

        // Check positive depth
        const float& PcZ = Pc.at<float>(2);
        const float invz = 1.0f / PcZ;
        if (PcZ < 0.0f)
            return true;

        cv::Point2f kp = mpCamera->project(Pc);
        for(int j=-5;j<=5;j++)
        {
            for(int i=-5;i<=5;i++)
            {
                if(kp.y+j<0||kp.y+j>mImMask.rows) continue;
                if(kp.x+i<0||kp.x+i>mImMask.cols) continue;
                if(mImMask.at<uchar>(kp.y + j, kp.x + i)!=0)
                    return false;
            }
        }
        return true;
    }
    void ComputeThreeMaxima(vector<int> histo, const int L, int& ind1, int& ind2, int& ind3)
    {
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;

        for (int i = 0; i < L; i++) {
            const int s = histo[i];
            if (s > max1) {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            } else if (s > max2) {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            } else if (s > max3) {
                max3 = s;
                ind3 = i;
            }
        }

        if (max2 < 0.1f * (float)max1) {
            ind2 = -1;
            ind3 = -1;
        } else if (max3 < 0.1f * (float)max1) {
            ind3 = -1;
        }
    }
    void Frame::UpdateMask(KeyFrame* pKF,cv::Mat imMask,cv::Mat &imFlow,cv::Mat &imFlowLast)
    {       
        if(imMask.empty())
            return;
        mImFlow=imFlow.clone();
        mnObject=pKF->mObjectNum;
        mnObjectOld=pKF->mObjectNum;
        mImObjectOld =  cv::Mat::zeros(imMask.size(),CV_8U);
        vector<int> rotHist = vector<int>(30,0);
        const float factor = 30 / 360.0f;
        for (int j = 0; j < imMask.rows; j++)
        {
            for (int k = 0; k < imMask.cols; k++)
            {
                int objectid=imMask.at<uchar>(j,k);
                const int flow_x = imFlow.at<cv::Vec2f>(j,k)[0];
                const int flow_y = imFlow.at<cv::Vec2f>(j,k)[1];
                if (objectid!=0)
                {
                    int pass_x = k+flow_x;
                    int pass_y = j+flow_y;
                    if(pass_x < imMask.cols && pass_x > 0 && pass_y < imMask.rows && pass_y > 0){
                        mImObjectOld.at<uchar>(pass_y,pass_x) = objectid;
                    }
                }
                else
                {
                    float rot = atan2(flow_y,flow_x);
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==30)
                        bin=0;
                    assert(bin>=0 && bin<30);
                    rotHist[bin]++;
                }
            }
        }
        cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3),cv::Point(-1,-1));
        cv::Mat kernel_open2 = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3),cv::Point(-1,-1));
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15), cv::Point(-1, -1));
        // // cv::Mat kernel_temp = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(5,5),cv::Point(-1,-1));

        cv::morphologyEx(mImObjectOld,mImMaskOld,cv::MORPH_CLOSE,kernel_open);
        // cv::morphologyEx(mImMaskOld,mImMaskOld,cv::MORPH_OPEN,kernel_open2);
        // cv::dilate(mImObject,mImObject,kernel_open);
        // cv::erode(mImObject,mImObject,kernel_open2);
        // cv::morphologyEx(mImObject,mImObject,cv::MORPH_OPEN,kernel_open2);
        // cv::imwrite("/out/debug/flowmask.png",mImMask>0);
        // cv::imwrite("/out/debug/flowmaskpre.png",imMask>0);
        cv::imwrite("/out/debug/"+to_string(mnId)+"flowmask.png",mImMaskOld>0);
        cv::imwrite("/out/debug/"+to_string(mnId)+"flowmaskpre.png",imMask>0);

        cv::dilate(mImMaskOld,mImMaskOld,kernel);
        ComputeThreeMaxima(rotHist,30,maxflow1,maxflow2,maxflow3);
        // cv::imwrite("/out/debug/flowmaskobject.png",mImObject>0);
        UpdatePrioriMovingProbability();
        mImObject=mImObjectOld.clone();
        mImMask=mImMaskOld.clone();

    }
   
    void Frame::UpdateMask(Frame* pKF,cv::Mat imMask,cv::Mat &imFlow,cv::Mat &imFlowLast)
    {       
        if(imMask.empty())
            return;
        mnObjectOld=pKF->mnObject;
        mnObject=pKF->mnObject;
        mImObjectOld =  cv::Mat::zeros(mImGray.size(),CV_8U);
        mImFlow = imFlow.clone();
        vector<int> rotHist = vector<int>(30,0);
        const float factor = 30 / 360.0f;
        for (int j = 0; j < imMask.rows; j++)
        {
            for (int k = 0; k < imMask.cols; k++)
            {
                int objectid=imMask.at<uchar>(j,k);
                const int flow_x = imFlow.at<cv::Vec2f>(j,k)[0];
                const int flow_y = imFlow.at<cv::Vec2f>(j,k)[1];
                if (objectid!=0)
                {
                    if(k+flow_x < imMask.cols && k+flow_x > 0 && j+flow_y < imMask.rows && j+flow_y > 0){
                        mImObjectOld.at<uchar>(j+flow_y,k+flow_x) = objectid;
                    }
                }
                else
                {
                    float rot = atan2(flow_y,flow_x);
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==30)
                        bin=0;
                    assert(bin>=0 && bin<30);
                    rotHist[bin]++;
                }
            }
        }
        cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3),cv::Point(-1,-1));
        cv::Mat kernel_open2 = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3),cv::Point(-1,-1));
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(15,15),cv::Point(-1,-1));
        cv::morphologyEx(mImObjectOld,mImMaskOld,cv::MORPH_CLOSE,kernel_open);
        // cv::morphologyEx(mImMaskOld,mImMaskOld,cv::MORPH_OPEN,kernel_open2);
        cv::imwrite("/out/debug/"+to_string(mnId)+"flowmask.png",mImMaskOld>0);
        cv::imwrite("/out/debug/"+to_string(mnId)+"flowmaskpre.png",imMask>0);
        cv::dilate(mImMaskOld,mImMaskOld,kernel);
        ComputeThreeMaxima(rotHist,30,maxflow1,maxflow2,maxflow3);
        mImObject=mImObjectOld.clone();
        mImMask=mImMaskOld.clone();
        UpdatePrioriMovingProbability();
    }
    
    void Frame::UpdateMask(Frame* pKF,Frame* pLastFrame,cv::Mat &imFlow,cv::Mat &imFlowLast)
    {       
        if(pLastFrame->mImMask.empty())
            return;
        // cout<<pKF->mnId<<" "<<mnId<<endl;
        mImFlow=imFlow.clone();
        cv::Mat imObjectTemp =  cv::Mat::zeros(pLastFrame->mImMask.size(),CV_8U);
        vector<int> rotHist = vector<int>(30,0);
        const float factor = 30 / 360.0f;
        for (int j = 0; j < imObjectTemp.rows; j++)
        {
            for (int k = 0; k < imObjectTemp.cols; k++)
            {
                int objectid=pLastFrame->mImObject.at<uchar>(j,k);
                const int flow_x = imFlow.at<cv::Vec2f>(j,k)[0];
                const int flow_y = imFlow.at<cv::Vec2f>(j,k)[1];
                if (objectid!=0)
                {
                    if(k+flow_x < imObjectTemp.cols && k+flow_x > 0 && j+flow_y < imObjectTemp.rows && j+flow_y > 0){
                        imObjectTemp.at<uchar>(j+flow_y,k+flow_x) = objectid;
                    }
                }
                else
                {
                    float rot = atan2(flow_y,flow_x);
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==30)
                        bin=0;
                    assert(bin>=0 && bin<30);
                    rotHist[bin]++;
                }
            }
        }
        if (mImObject.rows != 0 && pLastFrame->mnObject > 0){
            vector<bool> check = vector<bool>(pLastFrame->mnObject, false);
            for (int i = 1; i <= mnObject; i++)
            {
                for (int j = 1; j <= pLastFrame->mnObject; j++)
                {
                    if (check[j - 1])
                        continue;
                    float iou = cv::countNonZero((mImObject == i) & (imObjectTemp == j));
                    float all_ = cv::countNonZero((mImObject == i));
                    if (all_ == 0 || iou / all_ > 0.3)
                    {
                        // pKF->mImMask|=pKF->mImMaskOld;
                        check[j - 1] = true;
                        break;
                    }
                }
            }
            cv::Mat temp_cp = cv::Mat::zeros(mImObject.size(), CV_8UC1);
            cv::Mat temp_ob = cv::Mat::zeros(mImObject.size(), CV_8UC1);
            int tempnum = mnObject;
            for (int i = 1; i <= pLastFrame->mnObject; i++)
            {
                if (!check[i - 1])
                {
                    tempnum++;
                    cv::Mat temp_old = cv::Mat::ones(mImMask.size(), CV_8UC1) * tempnum;
                    copyTo(temp_old, temp_ob, imObjectTemp == i);
                }
            }
            mnObject = tempnum;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
            temp_ob.copyTo(mImObject, temp_ob != 0);
            
            // cv::morphologyEx(mImObject,mImObject,cv::MORPH_CLOSE,kernel);
            cv::Mat kernel_d = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15), cv::Point(-1, -1));
            cv::morphologyEx(mImObject,mImObject,cv::MORPH_CLOSE,kernel);
            cv::morphologyEx(mImObject,mImObject,cv::MORPH_OPEN,kernel);
            cv::dilate(mImObject,mImMask,kernel_d);
        }
        else if(mImMask.empty())
        {
            mnObject=pLastFrame->mnObject;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(15,15),cv::Point(-1,-1));
            mImObject = imObjectTemp.clone();
            cv::dilate(mImObject,mImMask,kernel);
        }

        // cv::imwrite("/out/debug/flowmask.png",mImMask>0);
        ComputeThreeMaxima(rotHist,30,maxflow1,maxflow2,maxflow3);
        // cv::imwrite("/out/debug/flowmaskobjpre.png",imObjectTemp>0);
        mImObjectOld=mImObject.clone();
        mImMaskOld=mImMask.clone();
        UpdatePrioriMovingProbability();
    }
   
    void Frame::UpdateMaskAndOpticalFlow(KeyFrame* pKF,cv::Mat &imMask,cv::Mat &imFlow,cv::Mat &imFlowLast,vector<Cluster*> &vpCluster)
    {       
        mImFlow=imFlow.clone();
        mImMask =  cv::Mat::zeros(imMask.size(),CV_8U);
        int n=vpCluster.size();
        vector<bool> updateFlag(n,false);
        vector<cv::Point2f> ptlu(n,cv::Point2f(480.,640.));
        vector<cv::Point2f> ptrd(n,cv::Point2f(0.,0.));
        for (int j = 0; j < imMask.rows; j++)
        {
            for (int k = 0; k < imMask.cols; k++)
            {
                int objectid=imMask.at<uchar>(j,k);
                if (objectid!=0)
                {
                    const int flow_x = imFlow.at<cv::Vec2f>(j,k)[0];
                    const int flow_y = imFlow.at<cv::Vec2f>(j,k)[1];

                    if(k+flow_x < imMask.cols && k+flow_x > 0 && j+flow_y < imMask.rows && j+flow_y > 0){
                        mImMask.at<uchar>(j+flow_y,k+flow_x) = objectid;
                        ptlu[objectid-1].x=min((float)k+flow_x,ptlu[objectid-1].x);
                        ptlu[objectid-1].y=min((float)j+flow_y,ptlu[objectid-1].y);
                        ptrd[objectid-1].x=max((float)k+flow_x,ptrd[objectid-1].x);
                        ptrd[objectid-1].y=max((float)j+flow_y,ptrd[objectid-1].y);
                        updateFlag[objectid-1]=true;
                    }
                }
            }
        }
        for(int i=0;i<updateFlag.size();i++)
        {
            if(updateFlag[i]){
                vpCluster[i]->updateBbox(ptlu[i],ptrd[i]);
            }

        }
        // if(mImMask.rows!=0)
        // cv::imwrite("/out/debug/flowmaskpre.png",mImMask>0);
        // pKF->mImMask=mImMask;
        // camera flow
        /*
        int step=4;
        for (int x = 0; x < imMask.cols; x+=step)
        {
            for(int y=0;y<imMask.rows;y+=step){
                float flow_xe = imFlow.at<cv::Vec2f>(y,x)[0];
                float flow_ye = imFlow.at<cv::Vec2f>(y,x)[1];

                if(x+flow_xe < imMask.cols && y+flow_ye < imMask.rows)
                {
                    // backgroung
                    if (imMask.at<uchar>(y,x)==0)
                    {   
                        continue;
                    }
                    else            // object
                    {
                        // save correspondences
                        mvObjFlowNext.push_back(cv::Point2f(flow_xe,flow_ye));
                        mvObjCorres.push_back(cv::KeyPoint(x+flow_xe,y+flow_ye,0,0,0,-1));
                        // save pixel location
                        mvObjKeys.push_back(cv::KeyPoint(x,y,0,0,0,-1));
                    }
                }
            }
        } 
        // cv::imwrite("/out/debug/flowmask.png",mImMask>0);
        */
    }

    void Frame::UpdateMaskAndOpticalFlow(Frame* pKF,cv::Mat &imMask,cv::Mat &imFlow,cv::Mat &imFlowLast,vector<Cluster*> &vpCluster)
    {       
        mImFlow=imFlow.clone();
        mImMask =  cv::Mat::zeros(imMask.size(),CV_8U);
        int n=vpCluster.size();
        vector<bool> updateFlag(n,false);
        vector<cv::Point2f> ptlu(n,cv::Point2f(480.,640.));
        vector<cv::Point2f> ptrd(n,cv::Point2f(0.,0.));
        for (int j = 0; j < imMask.rows; j++)
        {
            for (int k = 0; k < imMask.cols; k++)
            {
                int objectid=imMask.at<uchar>(j,k);
                if (objectid!=0)
                {
                    const int flow_x = imFlow.at<cv::Vec2f>(j,k)[0];
                    const int flow_y = imFlow.at<cv::Vec2f>(j,k)[1];
                    if(k+flow_x < imMask.cols && k+flow_x > 0 && j+flow_y < imMask.rows && j+flow_y > 0){
                        mImMask.at<uchar>(j+flow_y,k+flow_x) = objectid;
                        ptlu[objectid-1].x=min((float)k+flow_x,ptlu[objectid-1].x);
                        ptlu[objectid-1].y=min((float)j+flow_y,ptlu[objectid-1].y);
                        ptrd[objectid-1].x=max((float)k+flow_x,ptrd[objectid-1].x);
                        ptrd[objectid-1].y=max((float)j+flow_y,ptrd[objectid-1].y);
                        updateFlag[objectid-1]=true;
                    }
                }
            }
        }
        
        for(int i=0;i<updateFlag.size();i++)
        {
            if(updateFlag[i]){
                vpCluster[i]->updateBbox(ptlu[i],ptrd[i]);
            }
        }
        // pKF->mImMask=mImMask;
        // camera flow
        int step=4;
        for (int x = 0; x < imMask.cols; x+=step)
        {
            for(int y=0;y<imMask.rows;y+=step){
                float flow_xe = imFlow.at<cv::Vec2f>(y,x)[0];
                float flow_ye = imFlow.at<cv::Vec2f>(y,x)[1];

                if(x+flow_xe < imMask.cols && y+flow_ye < imMask.rows)
                {
                    // backgroung
                    if (imMask.at<uchar>(y,x)==0)
                    {   
                        continue;
                    }
                    else            // object
                    {
                        // save correspondences
                        mvObjFlowNext.push_back(cv::Point2f(flow_xe,flow_ye));
                        mvObjCorres.push_back(cv::KeyPoint(x+flow_xe,y+flow_ye,0,0,0,-1));
                        // save pixel location
                        mvObjKeys.push_back(cv::KeyPoint(x,y,0,0,0,-1));
                    }
                }
            }
        } 
        // cv::imwrite("/out/debug/flowmask.png",mImMask>0);

    }

    void Frame::UpdatePrioriMovingProbability()
    {
        if(mImMask.rows==0) return;
        cv::Mat showFeature = this->mImRGB.clone();

        float p_zd_md = 0.9;
        float p_zs_md = 0.1;
        float p_zs_ms = 0.9;
        float p_zd_ms = 0.1;

        auto start = std::chrono::steady_clock::now();
        // cv::Mat Rcw = this->GetPose().rowRange(0, 3).colRange(0, 3);
        // cv::Mat tcw = this->GetPose().rowRange(0, 3).col(3);
        // cv::Mat Ow = -Rcw.t() * tcw;
        // Remove outliers of current keyframe
        // unique_lock<mutex> lock(mMutexFeatures);
        bool bIsMapPointExists = false;

        // unique_lock<mutex> lock(this->GetMap()->mMutexMapUpdate);
        cv::Mat mapPointIdx = cv::Mat::ones(mImGray.size(),CV_8UC1)*-1;
        unordered_map<int,pair<float,cv::Mat>> vMvProb;
        for (int i = 0; i < this->N; i++) {
            // mark dynamic features
            cv::KeyPoint kp = this->mvKeys[i];
            if (kp.pt.x <= 0 || kp.pt.x >= this->mImMask.cols)
                continue;
            if (kp.pt.y <= 0 || kp.pt.y >= this->mImMask.rows)
                continue;

            MapPoint* pMP = this->mvpMapPoints[i];
            bIsMapPointExists = false;
            if (pMP) {
                if (!pMP->isBad())
                    bIsMapPointExists = true;
            }
            if (this->mImMask.at<uchar>((int)kp.pt.y, (int)kp.pt.x) != 0) {
                // this->mbIsHasDynamicObject = true;
                // dynamic object exists
                // visualization
                cv::circle(showFeature, kp.pt, 2, cv::Scalar(0, 0, 255), -1);
                this->mvbKptOutliers[i] = true;
                // this->mnDynamicPoints++;
                // if (bIsMapPointExists) {
                //     pMP->mnObservedDynamic++;
                //     // update moving probability
                //     // pMP->mMovingProbability = 1;
                // }
            } else {
                // visualization
                cv::circle(showFeature, kp.pt, 2, cv::Scalar(255, 0, 0), -1);
                this->mvbKptOutliers[i] = false;
                // if (bIsMapPointExists) {
                //     // pMP->mMovingProbability = 0;
                //     pMP->mnObservedStatic++;
                // }
            }
            if (bIsMapPointExists) {
                // update moving probability
                float p_old_d = pMP->GetMovingProbability();
                float p_old_s = 1 - p_old_d;
                float p_d,p_s,eta;
                if (this->mvbKptOutliers[i]) {
                    p_d = p_zd_md * p_old_d;
                    p_s = p_zd_ms * p_old_s;
                    eta = 1 / (p_d + p_s);
                    // pMP->mMovingProbability = eta * p_d;
                    pMP->SetMovingProbability(eta * p_d);
                } else {
                    p_d = p_zs_md * p_old_d;
                    p_s = p_zs_ms * p_old_s;
                    eta = 1 / (p_d + p_s);
                    // pMP->mMovingProbability = eta * p_d;
                    pMP->SetMovingProbability(eta * p_d);
                }
                /*cv::Mat x3D = this->UnprojectStereoCamera(i);
                if(x3D.rows!=0){
                    vMvProb[i]=make_pair(eta*p_d,x3D);
                    mapPointIdx.at<uchar>((int)kp.pt.y, (int)kp.pt.x)=i;
                }*/
            }

        } //end for
        /*
        float r=0.5;
        unordered_map<int,float> probUpdate;
        for (int i = 0; i < this->N; i++) 
        {
            cv::Point pt = this->mvKeys[i].pt;
            float fProb=vMvProb[i].first;
            if(fProb>0.5){
                float yita=0;
                float prefix=0; 
                cv::Mat x3D = vMvProb[i].second;
                for(int u=-5;u<=5;u++)
                {
                    for(int v=-5;v<=5;v++)
                    {                        
                        if (pt.x+v <= 0 || pt.x+v >= this->mImMask.cols)
                            continue;
                        if (pt.y+u <= 0 || pt.y+u >= this->mImMask.rows)
                            continue;
                        int idx=mapPointIdx.at<uchar>((int)pt.y+u, (int)pt.x+v);
                        if(idx!=-1&&vMvProb.count(idx)&&vMvProb[idx].first!=0)
                        {
                            float dis = norm(vMvProb[idx].second-x3D);
                            if(dis<r){
                                yita++;
                                if(vMvProb[idx].first>0.5)
                                    prefix+=1;
                                else
                                    prefix+=max(float(0),1-dis/r);
                            }
                        }
                    }
                }
                probUpdate[i]=0.5+prefix/yita*(fProb-0.5);
            }
        }
        unordered_map<int,float>::iterator it;
        for(it=probUpdate.begin();it!=probUpdate.end();it++)
        {
            MapPoint* pMP = this->mvpMapPoints[it->first];
            pMP->SetMovingProbability(it->second);
        }
        */
        // if(Semantic::GetInstance()->mbSaveResult)
            // Config::GetInstance()->saveImage(showFeature, "feature", "semantic_" + std::to_string(this->mnId) + ".png");
    }
    
}