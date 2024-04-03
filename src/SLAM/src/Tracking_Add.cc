#include "Tracking.h"
#include "ObjectDetection.h"
#include "MapObject.h"
#include "Converter.h"
#include "ORBmatcher.h"
using namespace std;
namespace ORB_SLAM3
{
    void Tracking::TrackClusterMapPoints()
    {
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        vector<Cluster*> vpMapCluster;
        vector<MapPoint*> vpMapClusterMapPoint;

        int nmatches = matcher.SearchClusterByBoW(&mLastFrame, mCurrentFrame);

    }
    
    void Tracking::ProcessMovingObject(const cv::Mat &imgray, cv::Mat &imggraypre,cv::Mat &outMask)
    {
        outMask = cv::Mat::zeros(imgray.size(),CV_8UC1);
        // Clear the previous data
        std::vector<cv::Point2f> F_prepoint,F_nextpoint,F2_prepoint,F2_nextpoint,prepoint, nextpoint;
        
        std::vector<uchar> state;
        std::vector<float> err;
        // T_M.clear();

        // Detect dynamic target and ultimately optput the T matrix
        
        cv::goodFeaturesToTrack(imggraypre, prepoint, 1000, 0.01, 8, cv::Mat(), 3, true, 0.04);
        cv::cornerSubPix(imggraypre, prepoint, cv::Size(10, 10), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
        cv::calcOpticalFlowPyrLK(imggraypre, imgray, prepoint, nextpoint, state, err, cv::Size(22, 22), 5, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));

        for (int i = 0; i < state.size(); i++)
        {
            if(state[i] != 0)
            {
                // int dx[10] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
                // int dy[10] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
                // int x1 = prepoint[i].x, y1 = prepoint[i].y;
                // int x2 = nextpoint[i].x, y2 = nextpoint[i].y;
                // if ((x1 < 5 || x1 >= imgray.cols - 5 || x2 < 5 || x2 >= imgray.cols - 5
                // || y1 < 5 || y1 >= imgray.rows - 5 || y2 < 5 || y2 >= imgray.rows - 5))
                // {
                //     state[i] = 0;
                //     continue;
                // }
                // double sum_check = 0;
                // for (int j = 0; j < 9; j++)
                //     sum_check += abs(imggraypre.at<uchar>(y1 + dy[j], x1 + dx[j]) - imgray.at<uchar>(y2 + dy[j], x2 + dx[j]));
                // if (sum_check > 50) state[i] = 0;
                if (state[i])
                {
                    F_prepoint.push_back(prepoint[i]);
                    F_nextpoint.push_back(nextpoint[i]);
                }
            }
        }
        if(F_prepoint.size()<10)
            return;
        // F-Matrix
        cv::Mat mask = cv::Mat(cv::Size(1, 300), CV_8UC1);
        cv::Mat F = cv::findFundamentalMat(F_prepoint, F_nextpoint, mask, cv::FM_RANSAC, 0.1, 0.99);
        for (int i = 0; i < mask.rows; i++)
        {
            if (mask.at<uchar>(i, 0) == 0);
            else
            {
                // Circle(pre_frame, F_prepoint[i], 6, Scalar(255, 255, 0), 3);
                double A = F.at<double>(0, 0)*F_prepoint[i].x + F.at<double>(0, 1)*F_prepoint[i].y + F.at<double>(0, 2);
                double B = F.at<double>(1, 0)*F_prepoint[i].x + F.at<double>(1, 1)*F_prepoint[i].y + F.at<double>(1, 2);
                double C = F.at<double>(2, 0)*F_prepoint[i].x + F.at<double>(2, 1)*F_prepoint[i].y + F.at<double>(2, 2);
                double dd = fabs(A*F_nextpoint[i].x + B*F_nextpoint[i].y + C) / sqrt(A*A + B*B); //Epipolar constraints
                if (dd <= 0.1)
                {
                    F2_prepoint.push_back(F_prepoint[i]);
                    F2_nextpoint.push_back(F_nextpoint[i]);
                }
            }
        }
        F_prepoint = F2_prepoint;
        F_nextpoint = F2_nextpoint;

        for (int i = 0; i < prepoint.size(); i++)
        {
            if (state[i] != 0)
            {
                double A = F.at<double>(0, 0)*prepoint[i].x + F.at<double>(0, 1)*prepoint[i].y + F.at<double>(0, 2);
                double B = F.at<double>(1, 0)*prepoint[i].x + F.at<double>(1, 1)*prepoint[i].y + F.at<double>(1, 2);
                double C = F.at<double>(2, 0)*prepoint[i].x + F.at<double>(2, 1)*prepoint[i].y + F.at<double>(2, 2);
                double dd = fabs(A*nextpoint[i].x + B*nextpoint[i].y + C) / sqrt(A*A + B*B);

                // Judge outliers
                if (dd <= 1) continue;
                for(int k=-6;k<=6;k++)
                {
                    for(int j=-6;j<=6;j++)
                    {
                        int x = nextpoint[i].x+k;
                        int y = nextpoint[i].y+j;
                        if(x<=0||y<=0||x>=outMask.cols||y>=outMask.rows) continue;
                        outMask.at<uchar>(y,x)=1;
                    }
                }
                // T_M.push_back(nextpoint[i]);
            }
        }
    }

    void Tracking::UpdateGeometryMovingProbability(cv::Mat &mask)
    {
        if(mask.rows==0) return;
        float p_zd_md = 0.75;
        float p_zs_md = 0.1;
        float p_zs_ms = 0.75;
        float p_zd_ms = 0.1;

        auto start = std::chrono::steady_clock::now();
        // cv::Mat Rcw = this->GetPose().rowRange(0, 3).colRange(0, 3);
        // cv::Mat tcw = this->GetPose().rowRange(0, 3).col(3);
        // cv::Mat Ow = -Rcw.t() * tcw;
        // Remove outliers of current keyframe
        // unique_lock<mutex> lock(mMutexFeatures);
        bool bIsMapPointExists = false;

        // unique_lock<mutex> lock(this->GetMap()->mMutexMapUpdate);
        for (int i = 0; i < mCurrentFrame.N; i++) {
            // mark dynamic features
            cv::KeyPoint kp = mCurrentFrame.mvKeys[i];
            if (kp.pt.x <= 0 || kp.pt.x >= mask.cols)
                continue;
            if (kp.pt.y <= 0 || kp.pt.y >= mask.rows)
                continue;

            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            bIsMapPointExists = false;
            if (pMP) {
                if (!pMP->isBad())
                    bIsMapPointExists = true;
            }

            if (mask.at<uchar>((int)kp.pt.y, (int)kp.pt.x) != 0) {
                // mCurrentFrame.mbIsHasDynamicObject = true;
                // dynamic object exists
                // visualization
                mCurrentFrame.mvbKptOutliers[i] = true;
                // mCurrentFrame.mnDynamicPoints++;
                // if (bIsMapPointExists) {
                //     pMP->mnObservedDynamic++;
                //     // update moving probability
                //     // pMP->mMovingProbability = 1;
                // }
            }
            if (bIsMapPointExists) {
                // update moving probability
                float p_old_d = pMP->GetMovingProbability();
                float p_old_s = 1 - p_old_d;

                if (mCurrentFrame.mvbKptOutliers[i]) {
                    float p_d = p_zd_md * p_old_d;
                    float p_s = p_zd_ms * p_old_s;
                    float eta = 1 / (p_d + p_s);
                    // pMP->mMovingProbability = eta * p_d;
                    pMP->SetMovingProbability(eta * p_d);
                } 
            }

        } //end for

        // this->GetMap()->IncreaseChangeIndex();
        // lock.unlock();

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        // if(Semantic::GetInstance()->mbSaveResult)
            Config::GetInstance()->saveImage(mask>0, "debug", "geomask.png");
    }

   
}