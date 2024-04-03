#ifndef _SEMANTIC_H_
#define _SEMANTIC_H_

#include "Common.h"
// based on ORB_SLAM3
#include "Atlas.h"
#include "KeyFrame.h"
#include "Tracking.h"
#include "Config.h"
#include "MapPoint.h"
#include "Map.h"
// semantic client
#include <actionlib/client/simple_action_client.h>
#include "semantic_cnn/SemanticCNNClient.h"

#include <opencv2/video.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <future>
using namespace semantic_cnn;
using namespace std;
void saveMat(cv::Mat mask, string path);
namespace ORB_SLAM3
{
    class semantic_wrapper;
    class Tracking;
    class Atlas;
    class MapPoint;
    class KeyFrame;
    class Map;

    class Semantic
    {
    public:
        // init
        static Semantic *GetInstance();
        static void SetConfigPath(string configPath);

        void IsEnableSemantic(bool bIsSemantic);
        void SetTracker(Tracking *pTracker);
        void SetAtlas(Atlas *pAtlas);
        // void SetSemanticObject();

        // about threads
        void Run();
        void SemanticTrackingThread();
        void SemanticBAThread();
        void SemanticThread();
        void RequestFinish();
        void SemanticThreadRe();
        void SemanticThreadReKey();

        // check dynamic map points
        bool IsDynamicMapPoint(const MapPoint *pMP);
        static void getBinMask(const cv::Mat &comMask, cv::Mat &binMask);

        // keyframe queue
        void InsertFrame(Frame *pKF);
        void InsertKeyFrame(KeyFrame *pKF);
        void InsertInitialKeyFrame(KeyFrame *pKF);
        void InsertSemanticRequest(KeyFrame *pKF);
        size_t GetLatestSemanticKeyFrame();
        void GetObjectOrder(KeyFrame *pKF, vector<int> &order);
        // KeyFrame* GetLatestSemanticKF(){return mpLatestKF;};
        void SetInitial(bool initial) { mbInitialKF = initial; }
        // time delay evaluation
        vector<float> mvTimeUpdateMovingProbability;
        vector<float> mvTimeMaskGeneration;
        vector<float> mvTimeSemanticOptimization;
        vector<size_t> mvSemanticDelay;
        static vector<string> mvstrLinesPath;

        static string mstrConfigPath;
        static int mbSaveResult;
        static bool mbViwer;
        static float mGrowThresh;
        static float mGrowThreshRGB;
        bool mbInitialKF = true;
        // release
        ~Semantic();

    private:
        // ============= Parameters ================
        static Semantic *mInstance;

        int mnLatestSemanticKeyFrameID;
        size_t mnTotalSemanticFrameNum;
        int mnBatchSize;

        // related ptr
        Tracking *mpTracker;
        Map *mpMap;
        Atlas *mpAtlas;

        KeyFrame* mpLatestKF=nullptr;
        Frame* mpLatestFrame=nullptr;

        future<cv::Mat> opticalFlowThread;
        cv::Mat mLastGray;
        cv::Mat mCurGray;
        cv::Mat mImObjectPre;
        int mnObjectPre=0;
        // independent on specific cnn method
        // string mstrCnnMethod;
        semantic_cnn::SemanticClient::Ptr mpSemanticCNN;
        cv::Ptr<cv::DenseOpticalFlow> DISoptFlow = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);

        // morphological filter
        int mnDilation_size;
        cv::Mat mKernel;

        // enable semantic moving probabilty or not
        bool mbIsSemantic;

        // semantic frame number
        int mnSemanticFrameNum;

        // moving probability threshold
        float mthDynamicThreshold;

        // thred and mutex
        condition_variable mConditionWaitResult;
        mutex mMutexResult;
        mutex mMutexNewSemanticRequest;
        mutex mMutexSemanticTrack;
        mutex mMutexNewKeyFrames;
        mutex mMutexNewFrames;
        mutex mMutexSemanticBA;
        mutex mMutexFinish;

        thread *mptSemantic;
        thread *mptSemanticBA;
        thread *mptSemanticTracking;

        // keyframe request list
        list<Frame *> mlNewFrames;
        list<KeyFrame *> mlNewKeyFrames;
        list<KeyFrame *> mlSemanticTrack;
        list<KeyFrame *> mlSemanticNew;
        list<KeyFrame *> mlSemanticBA;
        list<KeyFrame *> mlNewSemanticRequest;

        list<KeyFrame *> mlInitialKeyFrames;

        // dynamic objects used
        map<string, int> mmDynamicObjects;

        // main thread stop or not
        bool mbFinishRequest;

        // ============= Functions ================
        Semantic();

        // new semantic request
        bool CheckNewFrames();
        bool CheckNewKeyFrames();
        bool CheckNewSemanticRequest();
        bool CheckNewSemanticTrackRequest();
        bool CheckSemanticBARequest();

        void AddSemanticTrackRequest(KeyFrame *pKF);
        void AddSemanticBARequest(KeyFrame *pKF);
        // void GenerateMask(KeyFrame *pKF, const bool isDilate = true);
        void GenerateMask(KeyFrame *pKF, vector<cv::Mat> vLabel,
                          vector<cv::Mat> vObject, vector<int> vClassId,
                          vector<float> &mDepth, bool isDilate);
        void GenerateMask(KeyFrame *pKF, cv::Mat &imflow,vector<cv::Mat> vLabel,
                          vector<cv::Mat> vObject, vector<int> vClassId,
                          vector<float> &mDepth, bool isDilate);
        void GenerateMask(Frame *pKF, vector<cv::Mat> vLabel, vector<cv::Mat> vObject,
                                vector<int> vClassId, vector<float> &mDepth, bool isDilate);
        void GenerateMask(Frame *pKF, cv::Mat &imFlow,vector<cv::Mat> vLabel, vector<cv::Mat> vObject, vector<int> vClassId, vector<float> &mDepth, bool isDilate);
        // void GenerateMask(KeyFrame *pKF,vector<cv::Mat> vLabel, vector<cv::Mat> vObject, vector<int> vClassId, vector<float> &mDepth, bool isDilate);
        bool IsInImage(const float &x, const float &y, const cv::Mat &img);
        bool CheckFinish();
    };
}
#endif