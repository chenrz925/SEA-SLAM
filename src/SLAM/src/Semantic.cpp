#include "Semantic.h"
#include <opencv2/core/core.hpp>
#include "line_lbd/line_lbd_allclass.h"
#include "detect_3d_cuboid/detect_3d_cuboid.h"
#include "Converter.h"
using namespace std;
using namespace semantic_cnn;
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
void saveMat(cv::Mat mask, string path)
{
    cv::Mat I = mask;
    fstream PLeft48("/media/mxc/bs/SeaSLAM/" + path + ".txt", ios::out);
    if (!PLeft48.fail())
    {
        cout << "start writing PLeft48.txt" << endl;
        for (int i = 0; i < I.rows; i++)
        {
            for (int j = 0; j < I.cols; j++)
            {
                PLeft48 << int(I.at<uchar>(i, j)) << "\t";
            }
            PLeft48 << std::endl;
        }
        cout << "finish writing PLeft48.txt" << endl;
    }
    else
        cout << "can not open" << endl;
    PLeft48.close();
}
void pHashValue(Mat src, string &rst)
{
    Mat img, dst;
    double dIdex[64];
    double mean = 0.0;
    int k = 0;
    if (src.channels() == 3)
    {
        cvtColor(src, src, CV_BGR2GRAY);
        img = Mat_<double>(src);
    }
    else
    {
        img = Mat_<double>(src);
    }
    /* 第一步，缩放尺寸*/
    resize(img, img, Size(8, 8));
    /* 第二步，离散余弦变换，DCT系数求取*/
    dct(img, dst);
    /* 第三步，求取DCT系数均值（左上角8*8区块的DCT系数）*/
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            dIdex[k] = dst.at<double>(i, j);
            mean += dst.at<double>(i, j) / 64;
            ++k;
        }
    }
    /* 第四步，计算哈希值。*/
    for (int i = 0; i < 64; ++i)
    {
        if (dIdex[i] >= mean)
        {
            rst += '1';
        }
        else
        {
            rst += '0';
        }
    }
    // return rst;
}
int HanmingDistance(string &str1, string &str2)
{
    if ((str1.size() != 64) || (str2.size() != 64))
        return -1;
    int difference = 0;
    for (int i = 0; i < 64; i++)
    {
        if (str1[i] != str2[i])
            difference++;
    }
    return difference;
}

void getMaximizeImg(vector<bool> &used, vector<string> vstr, int n)
{
    map<int, set<int>> score;
    for (int i = 0; i < used.size(); i++)
    {
        if (vstr[i] == "")
            continue;
        for (int j = i + 1; j < used.size(); j++)
        {
            if (vstr[j] == "")
                continue;

            score[HanmingDistance(vstr[i], vstr[j])].insert(i);
            score[HanmingDistance(vstr[i], vstr[j])].insert(j);
        }
    }
    map<int, set<int>>::reverse_iterator it;

    for (it = score.rbegin(); it != score.rend(); it++)
    {
        set<int> temp = it->second;
        set<int>::iterator its = temp.begin();
        for (; its != temp.end(); its++)
        {
            if (used[*its] == false)
            {
                used[*its] = true;
                n--;
            }
            if (n == 0)
                break;
        }
        if (n == 0)
            break;
    }
}
namespace ORB_SLAM3
{
    Semantic *Semantic::mInstance = nullptr;
    string Semantic::mstrConfigPath = "";
    int Semantic::mbSaveResult = 0;
    bool Semantic::mbViwer = false;
    float Semantic::mGrowThresh = 0.001;
    float Semantic::mGrowThreshRGB = 0.001;
    vector<string> Semantic::mvstrLinesPath;
    Semantic::Semantic()
    {
        // initialize params
        mnLatestSemanticKeyFrameID = 0;
        mnTotalSemanticFrameNum = 0;
        // read config path
        cv::FileStorage semanticSettings(mstrConfigPath.c_str(), cv::FileStorage::READ);
        if (!semanticSettings.isOpened())
        {
            cerr << "Failed to open semantic settings file at: " << mstrConfigPath << endl;
            exit(1);
        }

        // reserve space
        int reserveSpace = semanticSettings["space.reserve"];
        mvTimeUpdateMovingProbability.reserve(reserveSpace);
        mvTimeSemanticOptimization.reserve(reserveSpace);
        mvTimeMaskGeneration.reserve(reserveSpace);

        mnBatchSize = semanticSettings["image.batch_size"];
        mnSemanticFrameNum = semanticSettings["image.semantic_num"];
        mbFinishRequest = false;

        mnDilation_size = semanticSettings["image.dilation_size"];
        mKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                            cv::Size(2 * mnDilation_size + 1, 2 * mnDilation_size + 1),
                                            cv::Point(mnDilation_size, mnDilation_size));

        mthDynamicThreshold = semanticSettings["dynamic.threshold"];
        mGrowThresh = semanticSettings["image.growThresh"];
        mGrowThreshRGB = semanticSettings["image.growThreshRGB"];
        // set objects
        cv::FileNodeIterator iter;
        for (iter = semanticSettings["dynamic.object"].begin(); iter != semanticSettings["dynamic.object"].end(); iter++)
        {
            mmDynamicObjects[(*iter)["class_name"]] = (*iter)["class_id"];
        }
        mbIsSemantic = true;

        LOG(INFO) << "Create semantic instance and start semantic thread";
    }

    Semantic::~Semantic()
    {
        LOG(INFO) << "Deinit Semantic";
        if (mptSemantic->joinable())
            mptSemantic->join();

        // if (mptSemanticTracking->joinable())
        //     mptSemanticTracking->join();
    }

    void Semantic::Run()
    {
        mptSemantic = new thread(&Semantic::SemanticThreadReKey, this);
        // mptSemantic = new thread(&Semantic::SemanticThreadRe, this);
        // mptSemantic = new thread(&Semantic::SemanticThread, this);
        // mptSemanticTracking = new thread(&Semantic::SemanticTrackingThread, this);
        // This thread seems not a must. I did not debug this thread in this sample code
        // This thread is described in the paper. Please try to enable it if you really need it.
        // mptSemanticBA = new std::thread(&Semantic::SemanticBAThread, this);
    }

    void Semantic::SemanticThread()
    {
        KeyFrame *pKF, *frontKF, *backKF;
        vector<cv::Mat> vmLabel;
        vector<cv::Mat> vmObject;
        vector<cv::Mat> vmRequest;
        vector<KeyFrame *> vKFs;
        vector<int> vBbox;
        vector<int> num;
        vector<int> vmClassId;
        size_t frontID, backID;
        // bool initial=true;

        // detect_3d_cuboid* detect_cuboid_obj = new detect_3d_cuboid(721.5377,721.5377,609.5593,172.854);
        // detect_cuboid_obj->whether_plot_final_images = true;
        // detect_cuboid_obj->whether_plot_detail_images = true;
        // detect_cuboid_obj->nominal_skew_ratio = 2;

        // line_lbd_detect line_lbd_obj;
        // line_lbd_obj.use_LSD = false;		// 使用 LSD 或 detector 线检测.
        // line_lbd_obj.line_length_thres = 15;  	// 去除较短的边线.

        vmLabel.reserve(mnBatchSize);
        vKFs.reserve(mnBatchSize);
        vmRequest.reserve(mnBatchSize);
        vmObject.reserve(mnBatchSize);
        num.reserve(mnBatchSize);

        // set cnn method
        LOG(INFO) << "Start Semantic  thread";
        mpSemanticCNN = make_shared<semantic_cnn::SemanticClient>(string("/semantic_server"), true);
        // this_thread::sleep_for(chrono::milliseconds(100));
        while (true)
        {
            if (!CheckNewKeyFrames())
            {
                if (!CheckFinish())
                {
                    this_thread::sleep_for(chrono::milliseconds(1));
                    continue;
                }
                else
                {
                    LOG(INFO) << "Semantic thread Stopping ....";
                    break;
                }
            }
            LOG(INFO) << "=================================================";
            std::cout << "=================================================" << endl;

            vmRequest.clear();
            vKFs.clear();
            vmLabel.clear();
            vmObject.clear();
            vBbox.clear();
            num.clear();
            vmClassId.clear();
            unique_lock<mutex> lock(mMutexNewKeyFrames);
            // int termi_kf=mlNewKeyFrames.size()/2;
            // int cnt_kf=0;
            for (std::list<KeyFrame *>::iterator it = mlNewKeyFrames.begin(); it != mlNewKeyFrames.end(); it++)
            {
                frontKF = *it;
                frontID = frontKF->mnId;
                // frontKF->SetNotErase();

                if (frontKF->IsSemanticReady())
                {
                    mnTotalSemanticFrameNum++;
                    AddSemanticTrackRequest(*it);
                    it = mlNewKeyFrames.erase(it);
                    continue;
                }
                vKFs.push_back(frontKF);
                vmRequest.push_back(frontKF->mImRGB);
                if (vKFs.size() == mnBatchSize / 2)
                {
                    break;
                }
            }

            LOG(INFO) << "frontID: " << frontID;

            // pick the latest keyframes
            // cnt_kf=-1;
            for (std::list<KeyFrame *>::reverse_iterator ir = mlNewKeyFrames.rbegin(); ir != mlNewKeyFrames.rend(); ir++)
            {
                backKF = *ir;
                backID = backKF->mnId;
                // cnt_kf++;
                // if(cnt_kf%termi_kf) continue;
                vKFs.push_back(backKF);
                vmRequest.push_back(backKF->mImRGB);
                if (vKFs.size() == mnBatchSize)
                {
                    // if(vKFs[mnBatchSize-1]->mnId<vKFs[mnBatchSize-2]->mnId){
                    //     swap(vKFs[mnBatchSize-1],vKFs[mnBatchSize-2]);
                    //     swap(vmRequest[mnBatchSize-1],vmRequest[mnBatchSize-2]);
                    // }
                    break;
                }
            }
            LOG(INFO) << "Back ID: " << backID;
            LOG(INFO) << "vKFS queue size: " << vKFs.size();
            // the minimum request size is mBatchSize
            if (vKFs.size() < mnBatchSize)
            {
                lock.unlock();
                vKFs.clear();
                vmRequest.clear();
                // LOG(WARNING) << "KeyFrame size is less than batch size";
                continue;
            }
            // key frame queue
            lock.unlock();

            // ask for semantic result
            mpSemanticCNN->Semantic(vmRequest, vmLabel, num, vmObject, vBbox, vmClassId);

            this_thread::sleep_for(chrono::milliseconds(50));
            // save semantic result and generate mask image
            if (num.size() != mnBatchSize)
            {
                LOG(ERROR) << "size of label is wrong";
            }
            int boxidx = 0, imgidx = 0;
            int j = 0;
            for (size_t i = 0; i < mnBatchSize; i++)
            {
                vector<cv::Mat> vObjectMask;
                vector<cv::Mat> vLabelMask;
                vector<int> vClassId;
                vector<float> meanDepth;
                // vKFs[i]->mImLabel = vmLabel[i];
                // vKFs[i]->mImObject = vmObject[i];

                vKFs[i]->mObjectNum = num[i];
                cv::Mat temp_img = vmRequest[i].clone();
                for (int k = 0; k < num[i]; k++, j++)
                {
                    // vector<int> temp;
                    cv::Rect temp;
                    temp.x = vBbox[j * 4];     //((box[0] + box[2]) / 2.);
                    temp.y = vBbox[j * 4 + 1]; //((box[1] + box[3]) / 2.);
                    temp.width = vBbox[j * 4 + 2] - vBbox[j * 4 + 0];
                    temp.height = vBbox[j * 4 + 3] - vBbox[j * 4 + 1];
                    vKFs[i]->mBbox.push_back(temp);
                    vClassId.push_back(vmClassId[imgidx]);
                    vObjectMask.push_back(vmObject[imgidx]);
                    vLabelMask.push_back(vmLabel[imgidx++]);
                }
                this->GenerateMask(vKFs[i], vLabelMask, vObjectMask, vClassId, meanDepth, true);
                // vector<int> order;
                // this->GetObjectOrder(vKFs[i],order);
                vKFs[i]->UpdatePrioriMovingProbability();
                vKFs[i]->InformSemanticReady(true);
                // if (vKFs[i]->mnFrameId > mnLatestSemanticKeyFrameID) {
                // if(i==mnBatchSize-1){
                //     mpTracker->SetSemanticMask(vKFs[i]);
                //     mpAtlas->useKF=true;
                // }
                mpTracker->SetSemanticMask(vKFs[i]);
                mpAtlas->useKF = true;
                // vKFs[i]->UpdatePrioriMovingProbability();
                // if(i==mnBatchSize-1){
                //     // // STEP 3.2 【边缘线检测】.
                //     // // @PARAM all_lines_raw 边5缘线存储的矩阵.
                //     // cv::Mat all_lines_mat;	// 检测到的线段信息，cv::Mat格式.
                //     // line_lbd_obj.detect_filter_lines(vKFs[i]->mImGray, all_lines_mat);

                //     // // 将 all_lines_mat 存储到 all_lines_raw 中.
                //     // Eigen::MatrixXd all_lines_raw(all_lines_mat.rows,4);		// TODO：4，线段的两个端点，每个点的xy坐标.
                //     // for (int rr=0;rr<all_lines_mat.rows;rr++)
                //     //         for (int cc=0;cc<4;cc++)
                //     //             all_lines_raw(rr,cc) = all_lines_mat.at<float>(rr,cc);
                //     cv::Mat poseKF = vKFs[i]->GetPoseInverse();
                //     Eigen::MatrixXd lines_all(100,4);
                //     read_all_number_txt(mvstrLinesPath[vKFs[i]->mnFrameId], lines_all);
                //     std::vector<ObjectSet> all_object_cuboids;
                //     detect_cuboid_obj->detect_cuboid(vKFs[i]->mImGray,Converter::toMatrix4d(poseKF) , vKFs[i]->mBbox, lines_all, all_object_cuboids);
                //     mpAtlas->UpdateClusters(vKFs[i],&mpTracker->mCurrentFrame,objectmask,all_object_cuboids);
                // }
                // mpTracker->UpdateObjectPoints(vKFs[i]);
                // Config::GetInstance()->saveImage(vKFs[i]->mImRGB,"keyframes",std::to_string(vKFs[i]->mnId)+ ".png");
                if (vKFs[i]->mnFrameId > mnLatestSemanticKeyFrameID)
                {
                    mnLatestSemanticKeyFrameID = vKFs[i]->mnFrameId;
                }
            }
            LOG(INFO) << "Size of semantic queue: " << mlNewKeyFrames.size();
            cout << "Size of semantic queue: " << mlNewKeyFrames.size() << endl;

            mbInitialKF = false;
        }
    }

    void Semantic::SemanticBAThread()
    {
        KeyFrame *pKF;
        LOG(INFO) << "Start Semantic BA thread";
        this_thread::sleep_for(chrono::milliseconds(50));
        while (true)
        {
            // new task is comming
            if (!CheckSemanticBARequest())
            {
                if (!CheckFinish())
                {
                    this_thread::sleep_for(chrono::milliseconds(1));
                    continue;
                } // check finish
                else
                {
                    cout << "Semantic BA thread stopping" << endl;
                    LOG(INFO) << "Semantic BA thread Stopping ....";
                    break;
                }
            }
            LOG(INFO) << "Size of semantic BA queue: " << mlSemanticBA.size();
            // cout << "Size of semantic BA queue: " << mlSemanticBA.size() << endl;
            // cout << "-----------Semantic BA:" << pKF->mnId << "-----------------" << endl;
            LOG(INFO) << "-----------Semantic BA:" << pKF->mnId << "------------------";
            unique_lock<mutex> lock(mMutexSemanticBA);
            pKF = mlSemanticBA.front();
            mlSemanticBA.pop_front();
            if (!pKF)
            {
                LOG(WARNING) << "Null key fame";
                lock.unlock();
                continue;
            }
            lock.unlock();

            if (pKF->mnId > 3)
            {
                // cout << "Optimize KF: " << pKF->mnId << endl;
                mpTracker->SemanticBA(pKF);
            }
        } // End while
        LOG(INFO) << "---------Semantic tracking thread finished---------------";
    }

    void Semantic::SemanticTrackingThread()
    {
        KeyFrame *lastKF;
        KeyFrame *currentKF;
        KeyFrame *latestKF;
        size_t lastProcessedId = 0;
        bool bSkip = false;
        int frameCount = -1;
        int lastSegID = -1;
        LOG(INFO) << "Start Semantic Tracking thread";
        this_thread::sleep_for(chrono::milliseconds(100));
        while (true)
        {
            if (!CheckNewSemanticTrackRequest())
            {
                if (!CheckFinish())
                {
                    this_thread::sleep_for(chrono::milliseconds(1));
                    continue;
                }
                else
                {
                    LOG(INFO) << "Semantic thread Stopping ....";
                    break;
                }
            }
            LOG(INFO) << "Size of semantic optimization queue: " << mlSemanticTrack.size();
            cout << "Size of semantic optimization queue: " << mlSemanticTrack.size() << endl;
            // if(mlSemanticTrack.size()<2) continue;
            unique_lock<mutex> lock(mMutexSemanticTrack);
            currentKF = mlSemanticTrack.front();
            mlSemanticTrack.pop_front();
            if (!currentKF)
            {
                LOG(WARNING) << "Null key fame";
                lock.unlock();
                continue;
            }
            lock.unlock();

            if (currentKF->mnId > 2)
            {
                cout << "Optimize KF: " << currentKF->mnId << endl;
                // Only optimize camera pose
                auto start = chrono::steady_clock::now();
                mpTracker->PoseOptimization(currentKF, true); // false
                auto end = chrono::steady_clock::now();
                chrono::duration<double> diff = end - start;
                mvTimeSemanticOptimization.push_back(diff.count());

                // optimize the map points at the same time
                // This maybe cause unstable of the system
                // mpTracker->PoseOptimization(currentKF, true);

                // [Debugging] Semantic BA
                // AddSemanticBARequest(currentKF);
            }
            // currentKF->SetErase();
            lastKF = currentKF;
            frameCount++;
            // End while
        }
        LOG(INFO) << "Semantic tracking thread finished";
    }

    bool Semantic::IsDynamicMapPoint(const MapPoint *pMP)
    {
        if (!pMP)
        {
            return true;
        }
        else if (pMP->mMovingProbability <= mthDynamicThreshold)
        {
            return false;
        }
        return true;
    }

    Semantic *Semantic::GetInstance()
    {
        if (mInstance == nullptr)
        {
            mInstance = new Semantic();
        }
        return mInstance;
    }
    void Semantic::SetConfigPath(string configPath)
    {
        mstrConfigPath = configPath;
    }
    bool Semantic::IsInImage(const float &x, const float &y, const cv::Mat &img)
    {
        return (x > 0 && x < img.cols && y > 0 && y < img.rows);
    }

    void Semantic::SetAtlas(Atlas *pAtlas)
    {
        mpAtlas = pAtlas;
        if (mpAtlas)
        {
            LOG(INFO) << "Set Atlas instance";
        }
    }

    void Semantic::SetTracker(Tracking *pTracker)
    {
        mpTracker = pTracker;
        if (mpTracker)
        {
            LOG(INFO) << "Set tracker instance";
        }
    }

    void Semantic::getBinMask(const cv::Mat &comMask, cv::Mat &binMask)
    {
        if (comMask.empty() || comMask.type() != CV_8UC1)
        {
            CV_Error(cv::Error::StsBadArg, "comMask is empty or type is not CV_8UC1");
        }
        if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
        {
            binMask.create(comMask.size(), CV_8UC1);
        }
        binMask = comMask & 1;
    }
    // check if any other thread ask for finsih semantic thread
    bool Semantic::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequest;
    }

    void Semantic::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequest = true;
        LOG(INFO) << "Semantic thread request stop";
        lock.unlock();

        this_thread::sleep_for(chrono::milliseconds(300));
        // if (mptSemantic->joinable())
        // {
        //     mptSemantic->join();
        // }
        // if (mptSemanticTracking->joinable())
        // {
        //     mptSemanticTracking->join();
        // }
    }

    bool Semantic::CheckNewSemanticRequest()
    {
        unique_lock<mutex> lock(mMutexNewSemanticRequest);
        bool res = (mlNewSemanticRequest.size() >= mnBatchSize) ? true : false;
        return res;
    }

    bool Semantic::CheckNewKeyFrames()
    {
        if (!mbIsSemantic)
        {
            return false;
        }
        unique_lock<mutex> lock(mMutexNewKeyFrames);
        bool res = (mlNewKeyFrames.size() >= mnBatchSize) ? true : false;
        // bool res_ini = (!mlInitialKeyFrames.empty()) ? true : false;
        return res; //||(res_ini&&mbInitialKF);
    }
    bool Semantic::CheckNewFrames()
    {
        if (!mbIsSemantic)
        {
            return false;
        }
        unique_lock<mutex> lock(mMutexNewFrames);
        bool res = (mlNewFrames.size() >= mnBatchSize) ? true : false;
        return res; 
    }
    bool Semantic::CheckNewSemanticTrackRequest()
    {
        unique_lock<mutex> lock(mMutexSemanticTrack);
        return !(mlSemanticTrack.empty());
    }

    bool Semantic::CheckSemanticBARequest()
    {
        unique_lock<mutex> lock(mMutexSemanticBA);
        return !(mlSemanticBA.empty()) && mlSemanticBA.size() > 5;
    }

    void Semantic::IsEnableSemantic(bool bIsSemantic)
    {
        mbIsSemantic = bIsSemantic;
    }

    void Semantic::AddSemanticBARequest(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexSemanticBA);
        mlSemanticBA.push_back(pKF);
    }

    void Semantic::AddSemanticTrackRequest(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexSemanticTrack);
        mlSemanticTrack.push_back(pKF);
    }

    void Semantic::InsertKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexNewKeyFrames);
        // pHashValue(pKF->mImGray, pKF->pHashV);
        mlNewKeyFrames.push_back(pKF);
    }
    void Semantic::InsertFrame(Frame *pKF)
    {
        unique_lock<mutex> lock(mMutexNewFrames);
        // pHashValue(pKF->mImGray, pKF->pHashV);
        mlNewFrames.push_back(pKF);
    }
    void Semantic::InsertInitialKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexNewKeyFrames);
        mlInitialKeyFrames.push_back(pKF);
    }

    void Semantic::InsertSemanticRequest(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexNewSemanticRequest);
        if (!pKF)
        {
            LOG(ERROR) << "pKF is null";
        }
        pKF->mbIsInsemanticQueue = true;

        LOG(INFO) << "Insert semantic request:" << pKF->mnId;
        mlNewSemanticRequest.push_back(pKF);
    }

    size_t Semantic::GetLatestSemanticKeyFrame()
    {
        return mnLatestSemanticKeyFrameID;
    }

    /*void Semantic::GenerateMask(KeyFrame *pKF, vector<cv::Mat> vLabel, vector<cv::Mat> vObject,
                                vector<int> vClassId, vector<float> &mDepth, bool isDilate)
    {
        // if (pKF->mImLabel.empty() || pKF->mImLabel.data == nullptr)
        // {
        //     LOG(WARNING) << "Generate Mask Failed :" << pKF->mnId;
        // }
        LOG(INFO) << "Generate Mask for Frame: " << pKF->mnId;
        if (pKF->mObjectNum == 0)
            return; // cv::Mat::zeros(pKF->mImGray.size(),CV_8UC1);
        auto start = chrono::steady_clock::now();

        cv::Mat mask = cv::Mat::zeros(pKF->mImLabel.size(), CV_8UC1);
        cv::Mat masktemp = cv::Mat::zeros(pKF->mImLabel.size(), CV_8UC1);
        map<string, int>::iterator iter;
        vector<double> meanDepth;
        meanDepth.resize(pKF->mObjectNum, 0);
        if (pKF->mImDepth.empty())
        {
            vector<int> cntKP(pKF->mObjectNum, 0);
            for (int i = 0; i < pKF->N; i++)
            {
                if (pKF->mvDepth[i] < 0)
                    continue;
                for (int j = 0; j < pKF->mObjectNum; j++)
                {
                    if (vObject[j].at<uchar>(pKF->mvKeys[i].pt.y, pKF->mvKeys[i].pt.x) != 0)
                    {
                        meanDepth[j] += pKF->mvDepth[i];
                        cntKP[j]++;
                    }
                }
            }
            for (int i = 0; i < pKF->mObjectNum; i++)
            {
                meanDepth[i] /= cntKP[i];
            }
        }
        else
        {
            // map<int,int> label2id;
            for (int i = 0; i < pKF->mObjectNum; i++)
            {
                cv::Mat depthMask = ((pKF->mImDepth > 0) & (vObject[i] != 0));
                cv::Mat curDepth;
                pKF->mImDepth.copyTo(curDepth, depthMask);
                meanDepth[i] = sum(curDepth)[0] / sum(depthMask / 255)[0];
                // label2id[vClassId[i]]=i+1;
            }
        }
        {

            pKF->mImLabel = vLabel[0];
            pKF->mImObject = vObject[0];

            // saveMat(vObject[0],"objectmask");
            for (int i = 1; i < pKF->mObjectNum; i++)
            {
                int curIdx = i;
                cv::Mat iou = ((pKF->mImLabel != 0) & (vLabel[i] != 0));
                if (cv::sum(iou)[0] != 0)
                {
                    for (int r = 0; r < pKF->mImLabel.rows; r++)
                    {
                        for (int c = 0; c < pKF->mImLabel.cols; c++)
                        {
                            if (iou.at<uchar>(r, c) != 0)
                            {
                                int preObjectID = pKF->mImObject.at<uchar>(r, c);
                                if (meanDepth[preObjectID] > meanDepth[i])
                                {
                                    pKF->mImObject.at<uchar>(r, c) = (i + 1);
                                    pKF->mImLabel.at<uchar>(r, c) = vClassId[i];
                                }
                            }
                        }
                    }
                }
                vLabel[i].copyTo(vLabel[i], (pKF->mImLabel == 0));
                vObject[i].copyTo(vObject[i], (pKF->mImObject == 0));
                pKF->mImLabel += vLabel[i];
                pKF->mImObject += vObject[i];
            }
        }

        mask = pKF->mImLabel.clone();
        mask = (mask > 0);

        pKF->mImLabel.copyTo(masktemp, mask);
        // dilate masks to filter out features on edge
        if (isDilate)
        {
            cv::dilate(masktemp, pKF->mImLabel, mKernel);
            if (pKF->mImLabel.empty())
            {
                LOG(INFO) << "Dilate operation failed, pKF mask is empty";
                return; // masktemp;
            }
        }
        else
        {
            masktemp.copyTo(pKF->mImLabel);
        }
        // mask.copyTo(pKF->mImMaskOld);
        masktemp = cv::Mat::zeros(pKF->mImLabel.size(), CV_8UC1);
        pKF->mImObject.copyTo(masktemp, mask);

        pKF->mImMask = pKF->mImObject.clone();
        // cv::dilate(backMask,pKF->mImMask,5);
        if (isDilate)
        {
            cv::dilate(masktemp, pKF->mImMask, mKernel);
            if (pKF->mImMask.empty())
            {
                LOG(INFO) << "Dilate operation failed, pKF mask is empty";
                return; // masktemp;
            }
        }
        else
        {
            masktemp.copyTo(pKF->mImMask);
        }
        // map<string, int>::iterator iterid;
        if (pKF->mImMaskOld.rows != 0 && pKF->mnObjectOld > 0)
        {
            Config::GetInstance()->saveImage(pKF->mImMask > 0, "mask", std::to_string(pKF->mnId) + "_" + std::to_string(pKF->mObjectNum) + ".png");
            // Config::GetInstance()->saveImage(pKF->mImMask, "mask",  "ori"+std::to_string(pKF->mnId)+"_"+std::to_string(pKF->mObjectNum)+ ".png");
            // cv::imwrite("/out/debug/maskpre.png",pKF->mImMaskOld>0);
            // cv::imwrite("/out/debug/objectmaskpre.png",pKF->mImObjectOld>0);
            vector<bool> check = vector<bool>(pKF->mnObjectOld, false);
            for (int i = 1; i <= pKF->mObjectNum; i++)
            {
                for (int j = 1; j <= pKF->mnObjectOld; j++)
                {
                    if (check[j - 1])
                        continue;
                    float iou = cv::countNonZero((pKF->mImObject == i) & (pKF->mImObjectOld == j));
                    float all_ = cv::countNonZero((pKF->mImObjectOld == j));
                    if (all_ == 0 || iou / all_ > 0.3)
                    {

                        // pKF->mImMask|=pKF->mImMaskOld;
                        check[j - 1] = true;
                        break;
                    }
                }
            }
            cv::Mat temp_cp = cv::Mat::zeros(masktemp.size(), CV_8UC1);
            cv::Mat temp_ob = cv::Mat::zeros(masktemp.size(), CV_8UC1);
            int tempnum = pKF->mObjectNum;
            for (int i = 1; i <= pKF->mnObjectOld; i++)
            {
                if (!check[i - 1])
                {
                    tempnum++;
                    cv::Mat temp_old = cv::Mat::ones(masktemp.size(), CV_8UC1) * tempnum;
                    copyTo(temp_old, temp_cp, pKF->mImMaskOld == i);
                    copyTo(temp_old, temp_ob, pKF->mImObjectOld == i);
                }
            }
            pKF->mObjectNum = tempnum;
            cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
            // cv::morphologyEx(temp_ob,temp_ob,cv::MORPH_CLOSE,kernel_open);
            // cv::morphologyEx(temp_ob,temp_ob,cv::MORPH_OPEN,kernel_open);
            cv::dilate(temp_ob, temp_ob, kernel_open);
            temp_cp.copyTo(pKF->mImMask, temp_cp != 0);
            temp_ob.copyTo(pKF->mImObject, temp_ob != 0);
            cv::erode(pKF->mImObject, pKF->mImObject, kernel_open);
            // cv::imwrite("/out/debug/maskpre.png",pKF->mImMaskOld>0);
            // Config::GetInstance()->saveImage(pKF->mImMask > 0, "mask_fuse", "fuse" + std::to_string(pKF->mnId) + "_" + std::to_string(pKF->mObjectNum) + ".png");
            // Config::GetInstance()->saveImage(pKF->mImMask, "mask_fuse", "orifuse" + std::to_string(pKF->mnId)+"_"+std::to_string(pKF->mObjectNum) + ".png");
            // Config::GetInstance()->saveImage(pKF->mImRGB, "keyframes", std::to_string(pKF->mnId) + ".png");
            // cv::imwrite("/out/debug/mask.png",pKF->mImMask>0);
            // cv::imwrite("/out/debug/objectmask.png",pKF->mImObject>0);
        }
        // pKF->mImMask = pKF->mImObject;
        // saveMat(pKF->mImObject,"object_mask"+to_string(pKF->mnId));
        // cv::imwrite("/out/debug/mask.png",pKF->mImMask>0);

        auto end = chrono::steady_clock::now();
        // saveMat(pKF->mImObject,"objectmask");
        // time consuming
        chrono::duration<double> diff = end - start;
        LOG(INFO) << "Time to update moving probability:" << setw(3) << diff.count() * 1000 << " ms";
        mvTimeMaskGeneration.push_back(diff.count());
        // return backMask;
    }*/
    void Semantic::GenerateMask(Frame *pKF, vector<cv::Mat> vLabel, vector<cv::Mat> vObject,
                                vector<int> vClassId, vector<float> &mDepth, bool isDilate)
    {
        // if (pKF->mImLabel.empty() || pKF->mImLabel.data == nullptr)
        // {
        //     LOG(WARNING) << "Generate Mask Failed :" << pKF->mnId;
        // }
        LOG(INFO) << "Generate Mask for Frame: " << pKF->mnId;
        if (pKF->mnObject == 0)
            return; // cv::Mat::zeros(pKF->mImGray.size(),CV_8UC1);
        int curObject = pKF->mnObject;
        auto start = chrono::steady_clock::now();

        cv::Mat mask = cv::Mat::zeros(pKF->mImGray.size(), CV_8UC1);
        cv::Mat masktemp = cv::Mat::zeros(pKF->mImGray.size(), CV_8UC1);
        map<string, int>::iterator iter;
        vector<double> meanDepth;
        meanDepth.resize(curObject, 0);
        if (pKF->mImDepth.empty())
        {
            vector<int> cntKP(curObject, 0);
            for (int i = 0; i < pKF->N; i++)
            {
                if (pKF->mvDepth[i] < 0)
                    continue;
                if(pKF->mvKeys[i].pt.y>=pKF->mImGray.rows||pKF->mvKeys[i].pt.y<=0)
                    continue;
                if(pKF->mvKeys[i].pt.x>=pKF->mImGray.cols||pKF->mvKeys[i].pt.x<=0)
                    continue;
                for (int j = 0; j < curObject; j++)
                {
                    if (vObject[j].at<uchar>(pKF->mvKeys[i].pt.y, pKF->mvKeys[i].pt.x) != 0)
                    {
                        meanDepth[j] += pKF->mvDepth[i];
                        cntKP[j]++;
                    }
                }
            }
            for (int i = 0; i < curObject; i++)
            {
                meanDepth[i] /= cntKP[i];
            }
        }
        else
        {
            // map<int,int> label2id;
            for (int i = 0; i < curObject; i++)
            {
                cv::Mat depthMask = ((pKF->mImDepth > 0) & (vObject[i] != 0));
                cv::Mat curDepth;
                pKF->mImDepth.copyTo(curDepth, depthMask);
                meanDepth[i] = sum(curDepth)[0] / sum(depthMask / 255)[0];
                // label2id[vClassId[i]]=i+1;
            }
        }
        {

            pKF->mImLabel = vLabel[0];
            pKF->mImObject = vObject[0];

            // saveMat(vObject[0],"objectmask");
            for (int i = 1; i < curObject; i++)
            {
                cv::Mat iou = ((pKF->mImLabel != 0) & (vLabel[i] != 0));
                if (cv::sum(iou)[0] != 0)
                {
                    for (int r = 0; r < pKF->mImLabel.rows; r++)
                    {
                        for (int c = 0; c < pKF->mImLabel.cols; c++)
                        {
                            if (iou.at<uchar>(r, c) != 0)
                            {
                                int preObjectID = pKF->mImObject.at<uchar>(r, c);
                                if (meanDepth[preObjectID] > meanDepth[i])
                                {
                                    pKF->mImObject.at<uchar>(r, c) = (i + 1);
                                    pKF->mImLabel.at<uchar>(r, c) = vClassId[i];
                                }
                            }
                        }
                    }
                }
                vLabel[i].copyTo(vLabel[i], (pKF->mImLabel == 0));
                vObject[i].copyTo(vObject[i], (pKF->mImObject == 0));
                pKF->mImLabel += vLabel[i];
                pKF->mImObject += vObject[i];
            }
        }

        mask = pKF->mImLabel.clone();
        mask = (mask > 0);

        pKF->mImLabel.copyTo(masktemp, mask);
        // dilate masks to filter out features on edge
        if (isDilate)
        {
            cv::dilate(masktemp, pKF->mImLabel, mKernel);
            if (pKF->mImLabel.empty())
            {
                LOG(INFO) << "Dilate operation failed, pKF mask is empty";
                return; // masktemp;
            }
        }
        else
        {
            masktemp.copyTo(pKF->mImLabel);
        }
        // mask.copyTo(pKF->mImMaskOld);
        masktemp = cv::Mat::zeros(pKF->mImLabel.size(), CV_8UC1);
        pKF->mImObject.copyTo(masktemp, mask);

        pKF->mImMask = pKF->mImObject.clone();
        // cv::dilate(backMask,pKF->mImMask,5);
        if (isDilate)
        {
            cv::dilate(masktemp, pKF->mImMask, mKernel);
            if (pKF->mImMask.empty())
            {
                LOG(INFO) << "Dilate operation failed, pKF mask is empty";
                return; // masktemp;
            }
        }
        else
        {
            masktemp.copyTo(pKF->mImMask);
        }
        // map<string, int>::iterator iterid;
        if (pKF->mImMaskOld.rows != 0 && pKF->mnObjectOld > 0)
        {
            // Config::GetInstance()->saveImage(pKF->mImMask > 0, "mask", std::to_string(pKF->mnId) + "_" + std::to_string(pKF->mnObject) + ".png");
            // Config::GetInstance()->saveImage(pKF->mImMask, "mask",  "ori"+std::to_string(pKF->mnId)+"_"+std::to_string(pKF->mObjectNum)+ ".png");
            // cv::imwrite("/out/debug/maskpre.png",pKF->mImMaskOld>0);
            // cv::imwrite("/out/debug/objectmaskpre.png",pKF->mImObjectOld>0);
            vector<bool> check = vector<bool>(pKF->mnObjectOld, false);
            for (int i = 1; i <= pKF->mnObject; i++)
            {
                for (int j = 1; j <= pKF->mnObjectOld; j++)
                {
                    if (check[j - 1])
                        continue;
                    float iou = cv::countNonZero((pKF->mImObject == i) & (pKF->mImObjectOld == j));
                    float all_ = cv::countNonZero((pKF->mImObjectOld == j));
                    if (all_ == 0 || iou / all_ > 0.3)
                    {

                        // pKF->mImMask|=pKF->mImMaskOld;
                        check[j - 1] = true;
                        break;
                    }
                }
            }
            cv::Mat temp_cp = cv::Mat::zeros(masktemp.size(), CV_8UC1);
            cv::Mat temp_ob = cv::Mat::zeros(masktemp.size(), CV_8UC1);
            int tempnum = pKF->mnObject;
            for (int i = 1; i <= pKF->mnObjectOld; i++)
            {
                if (!check[i - 1])
                {
                    tempnum++;
                    cv::Mat temp_old = cv::Mat::ones(masktemp.size(), CV_8UC1) * tempnum;
                    copyTo(temp_old, temp_cp, pKF->mImMaskOld == i);
                    copyTo(temp_old, temp_ob, pKF->mImObjectOld == i);
                }
            }
            pKF->mnObject = tempnum;
            cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
            // cv::morphologyEx(temp_ob,temp_ob,cv::MORPH_CLOSE,kernel_open);
            // cv::morphologyEx(temp_ob,temp_ob,cv::MORPH_OPEN,kernel_open);
            cv::dilate(temp_ob, temp_ob, kernel_open);
            temp_cp.copyTo(pKF->mImMask, temp_cp != 0);
            temp_ob.copyTo(pKF->mImObject, temp_ob != 0);
            cv::erode(pKF->mImObject, pKF->mImObject, kernel_open);
            // cv::imwrite("/out/debug/maskpre.png",pKF->mImMaskOld>0);
            // Config::GetInstance()->saveImage(pKF->mImMask > 0, "mask_fuse", "fuse" + std::to_string(pKF->mnId) + "_" + std::to_string(pKF->mObjectNum) + ".png");
            // Config::GetInstance()->saveImage(pKF->mImMask, "mask_fuse", "orifuse" + std::to_string(pKF->mnId)+"_"+std::to_string(pKF->mObjectNum) + ".png");
            // Config::GetInstance()->saveImage(pKF->mImRGB, "keyframes", std::to_string(pKF->mnId) + ".png");
            // cv::imwrite("/out/debug/mask.png",pKF->mImMask>0);
            // cv::imwrite("/out/debug/objectmask.png",pKF->mImObject>0);
        }
        // pKF->mImMask = pKF->mImObject;
        // saveMat(pKF->mImObject,"object_mask"+to_string(pKF->mnId));
        // cv::imwrite("/out/debug/mask.png",pKF->mImMask>0);
        // cv::imwrite("/out/debug/objectmask.png",pKF->mImObject>0);

        auto end = chrono::steady_clock::now();
        // saveMat(pKF->mImObject,"objectmask");
        // time consuming
        chrono::duration<double> diff = end - start;
        LOG(INFO) << "Time to update moving probability:" << setw(3) << diff.count() * 1000 << " ms";
        mvTimeMaskGeneration.push_back(diff.count());
        // return backMask;
    }
    
    void Semantic::GenerateMask(Frame *pKF, cv::Mat &imFlow,vector<cv::Mat> vLabel, vector<cv::Mat> vObject, vector<int> vClassId, vector<float> &mDepth, bool isDilate)
    {
        LOG(INFO) << "Generate Mask for Frame: " << pKF->mnId;
        if (pKF->mnObject == 0)
            return; // cv::Mat::zeros(pKF->mImGray.size(),CV_8UC1);
        int curObject = pKF->mnObject;
        auto start = chrono::steady_clock::now();

        cv::Mat mask = cv::Mat::zeros(pKF->mImGray.size(), CV_8UC1);
        cv::Mat masktemp = cv::Mat::zeros(pKF->mImGray.size(), CV_8UC1);
        map<string, int>::iterator iter;
        vector<double> meanDepth;
        meanDepth.resize(curObject, 0);
        pKF->mImFlow = imFlow.clone();
        if (pKF->mImDepth.empty())
        {
            vector<int> cntKP(curObject, 0);
            for (int i = 0; i < pKF->N; i++)
            {
                if (pKF->mvDepth[i] < 0)
                    continue;
                if(pKF->mvKeys[i].pt.y>=pKF->mImGray.rows||pKF->mvKeys[i].pt.y<=0)
                    continue;
                if(pKF->mvKeys[i].pt.x>=pKF->mImGray.cols||pKF->mvKeys[i].pt.x<=0)
                    continue;
                for (int j = 0; j < curObject; j++)
                {
                    if (vObject[j].at<uchar>(pKF->mvKeys[i].pt.y, pKF->mvKeys[i].pt.x) != 0)
                    {
                        meanDepth[j] += pKF->mvDepth[i];
                        cntKP[j]++;
                    }
                }
            }
            for (int i = 0; i < curObject; i++)
            {
                meanDepth[i] /= cntKP[i];
            }
        }
        else
        {
            // map<int,int> label2id;
            for (int i = 0; i < curObject; i++)
            {
                cv::Mat depthMask = ((pKF->mImDepth > 0) & (vObject[i] != 0));
                cv::Mat curDepth;
                pKF->mImDepth.copyTo(curDepth, depthMask);
                meanDepth[i] = sum(curDepth)[0] / sum(depthMask / 255)[0];
                // label2id[vClassId[i]]=i+1;
            }
        }
        {

            pKF->mImLabel = vLabel[0];
            pKF->mImObject = vObject[0];

            // saveMat(vObject[0],"objectmask");
            for (int i = 1; i < curObject; i++)
            {
                cv::Mat iou = ((pKF->mImLabel != 0) & (vLabel[i] != 0));
                if (cv::sum(iou)[0] != 0)
                {
                    for (int r = 0; r < pKF->mImLabel.rows; r++)
                    {
                        for (int c = 0; c < pKF->mImLabel.cols; c++)
                        {
                            if (iou.at<uchar>(r, c) != 0)
                            {
                                int preObjectID = pKF->mImObject.at<uchar>(r, c);
                                if (meanDepth[preObjectID] > meanDepth[i])
                                {
                                    pKF->mImObject.at<uchar>(r, c) = (i + 1);
                                    pKF->mImLabel.at<uchar>(r, c) = vClassId[i];
                                }
                            }
                        }
                    }
                }
                vLabel[i].copyTo(vLabel[i], (pKF->mImLabel == 0));
                vObject[i].copyTo(vObject[i], (pKF->mImObject == 0));
                pKF->mImLabel += vLabel[i];
                pKF->mImObject += vObject[i];
            }
        }

        mask = pKF->mImLabel.clone();
        mask = (mask > 0);

        pKF->mImLabel.copyTo(masktemp, mask);
        // dilate masks to filter out features on edge
        if (isDilate)
        {
            cv::dilate(masktemp, pKF->mImLabel, mKernel);
            if (pKF->mImLabel.empty())
            {
                LOG(INFO) << "Dilate operation failed, pKF mask is empty";
                return; // masktemp;
            }
        }
        else
        {
            masktemp.copyTo(pKF->mImLabel);
        }
        // mask.copyTo(pKF->mImMaskOld);
        masktemp = cv::Mat::zeros(pKF->mImLabel.size(), CV_8UC1);
        pKF->mImObject.copyTo(masktemp, mask);

        pKF->mImMask = pKF->mImObject.clone();
        // cv::dilate(backMask,pKF->mImMask,5);
        if (isDilate)
        {
            cv::dilate(masktemp, pKF->mImMask, mKernel);
            if (pKF->mImMask.empty())
            {
                LOG(INFO) << "Dilate operation failed, pKF mask is empty";
                return; // masktemp;
            }
        }
        else
        {
            masktemp.copyTo(pKF->mImMask);
        }
        // Config::GetInstance()->saveImage(pKF->mImMask > 0, "debug",  "ori_mask.png");


        cv::Mat imMaskOld = cv::Mat::zeros(pKF->mImMask.size(),CV_8UC1);
        if(mnObjectPre>0){
            int nObjectOld = mnObjectPre;
            vector<int> rotHist = vector<int>(30,0);
            const float factor = 30 / 360.0f;
            for (int j = 0; j < imMaskOld.rows; j++)
            {
                for (int k = 0; k < imMaskOld.cols; k++)
                {
                    int objectid=mImObjectPre.at<uchar>(j,k);
                    const int flow_x = imFlow.at<cv::Vec2f>(j,k)[0];
                    const int flow_y = imFlow.at<cv::Vec2f>(j,k)[1];
                    if (objectid!=0)
                    {
                        if(k+flow_x < imMaskOld.cols && k+flow_x > 0 && j+flow_y < imMaskOld.rows && j+flow_y > 0){
                            imMaskOld.at<uchar>(j+flow_y,k+flow_x) = objectid;
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
            ComputeThreeMaxima(rotHist,30,pKF->maxflow1,pKF->maxflow2,pKF->maxflow3);
            
            vector<bool> check = vector<bool>(nObjectOld, false);
            for (int i = 1; i <= curObject; i++)
            {
                for (int j = 1; j <= nObjectOld; j++)
                {
                    if (check[j - 1])
                        continue;
                    float iou = cv::countNonZero((pKF->mImObject == i) & (imMaskOld == j));
                    float all_ = cv::countNonZero((imMaskOld == j));
                    if (all_ == 0 || iou / all_ > 0.3)
                    {

                        // pKF->mImMask|=pKF->mImMaskOld;
                        check[j - 1] = true;
                        break;
                    }
                }
            }
            cv::Mat temp_cp = cv::Mat::zeros(masktemp.size(), CV_8UC1);
            cv::Mat temp_ob = cv::Mat::zeros(masktemp.size(), CV_8UC1);
            int tempnum = curObject;
            for (int i = 1; i <= nObjectOld; i++)
            {
                if (!check[i - 1])
                {
                    tempnum++;
                    cv::Mat temp_old = cv::Mat::ones(masktemp.size(), CV_8UC1) * tempnum;
                    copyTo(temp_old, temp_ob, imMaskOld == i);
                }
            }
            pKF->mnObject = tempnum;
            cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
            cv::morphologyEx(temp_ob,temp_ob,cv::MORPH_CLOSE,kernel_open);
            // cv::dilate(temp_ob, temp_ob, kernel_open);
            cv::dilate(temp_ob,temp_cp,mKernel);
            temp_cp.copyTo(pKF->mImMask, temp_cp != 0);
            temp_ob.copyTo(pKF->mImObject, temp_ob != 0);
            cv::erode(pKF->mImObject, pKF->mImObject, kernel_open);
            // cv::imwrite("/out/debug/maskpre.png",pKF->mImMaskOld>0);
        }
        // Config::GetInstance()->saveImage(pKF->mImMask > 0, "debug",  "ori_mask.png");

        // Config::GetInstance()->saveImage(pKF->mImMask > 0, "debug", "fuse_mask.png");
        // Config::GetInstance()->saveImage(pKF->mImRGB, "keyframes","rgb.png");
        // pKF->mImMask = pKF->mImObject;
        // saveMat(pKF->mImObject,"object_mask"+to_string(pKF->mnId));
        // cv::imwrite("/out/debug/mask.png",pKF->mImMask>0);
        // cv::imwrite("/out/debug/objectmask.png",pKF->mImObject>0);
        
        auto end = chrono::steady_clock::now();
        // saveMat(pKF->mImObject,"objectmask");
        // time consuming
        chrono::duration<double> diff = end - start;
        LOG(INFO) << "Time to update moving probability:" << setw(3) << diff.count() * 1000 << " ms";
        mvTimeMaskGeneration.push_back(diff.count());
        // return backMask;
    }
    
    void Semantic::GenerateMask(KeyFrame *pKF,vector<cv::Mat> vLabel, vector<cv::Mat> vObject, vector<int> vClassId, vector<float> &mDepth, bool isDilate)
    {
        LOG(INFO) << "Generate Mask for Frame: " << pKF->mnId;
        if (pKF->mObjectNum == 0)
            return; // cv::Mat::zeros(pKF->mImGray.size(),CV_8UC1);
        int curObject = pKF->mObjectNum;
        auto start = chrono::steady_clock::now();

        cv::Mat mask = cv::Mat::zeros(pKF->mImGray.size(), CV_8UC1);
        cv::Mat masktemp = cv::Mat::zeros(pKF->mImGray.size(), CV_8UC1);
        map<string, int>::iterator iter;
        vector<double> meanDepth;
        meanDepth.resize(curObject, 0);
        if (pKF->mImDepth.empty())
        {
            vector<int> cntKP(curObject, 0);
            for (int i = 0; i < pKF->N; i++)
            {
                if (pKF->mvDepth[i] < 0)
                    continue;
                if(pKF->mvKeys[i].pt.y>=pKF->mImGray.rows||pKF->mvKeys[i].pt.y<=0)
                    continue;
                if(pKF->mvKeys[i].pt.x>=pKF->mImGray.cols||pKF->mvKeys[i].pt.x<=0)
                    continue;
                for (int j = 0; j < curObject; j++)
                {
                    if (vObject[j].at<uchar>(pKF->mvKeys[i].pt.y, pKF->mvKeys[i].pt.x) != 0)
                    {
                        meanDepth[j] += pKF->mvDepth[i];
                        cntKP[j]++;
                    }
                }
            }
            for (int i = 0; i < curObject; i++)
            {
                meanDepth[i] /= cntKP[i];
            }
        }
        else
        {
            // map<int,int> label2id;
            for (int i = 0; i < curObject; i++)
            {
                cv::Mat depthMask = ((pKF->mImDepth > 0) & (vObject[i] != 0));
                cv::Mat curDepth;
                pKF->mImDepth.copyTo(curDepth, depthMask);
                meanDepth[i] = sum(curDepth)[0] / sum(depthMask / 255)[0];
                // label2id[vClassId[i]]=i+1;
            }
        }
        {

            pKF->mImLabel = vLabel[0];
            pKF->mImObject = vObject[0];

            // saveMat(vObject[0],"objectmask");
            for (int i = 1; i < curObject; i++)
            {
                cv::Mat iou = ((pKF->mImLabel != 0) & (vLabel[i] != 0));
                if (cv::sum(iou)[0] != 0)
                {
                    for (int r = 0; r < pKF->mImLabel.rows; r++)
                    {
                        for (int c = 0; c < pKF->mImLabel.cols; c++)
                        {
                            if (iou.at<uchar>(r, c) != 0)
                            {
                                int preObjectID = pKF->mImObject.at<uchar>(r, c);
                                if (meanDepth[preObjectID] > meanDepth[i])
                                {
                                    pKF->mImObject.at<uchar>(r, c) = (i + 1);
                                    pKF->mImLabel.at<uchar>(r, c) = vClassId[i];
                                }
                            }
                        }
                    }
                }
                vLabel[i].copyTo(vLabel[i], (pKF->mImLabel == 0));
                vObject[i].copyTo(vObject[i], (pKF->mImObject == 0));
                pKF->mImLabel += vLabel[i];
                pKF->mImObject += vObject[i];
            }
        }

        mask = pKF->mImLabel.clone();
        mask = (mask > 0);

        pKF->mImLabel.copyTo(masktemp, mask);
        // dilate masks to filter out features on edge
        if (isDilate)
        {
            cv::dilate(masktemp, pKF->mImLabel, mKernel);
            if (pKF->mImLabel.empty())
            {
                LOG(INFO) << "Dilate operation failed, pKF mask is empty";
                return; // masktemp;
            }
        }
        else
        {
            masktemp.copyTo(pKF->mImLabel);
        }
        // mask.copyTo(pKF->mImMaskOld);
        masktemp = cv::Mat::zeros(pKF->mImLabel.size(), CV_8UC1);
        pKF->mImObject.copyTo(masktemp, mask);

        pKF->mImMask = pKF->mImObject.clone();
        // cv::dilate(backMask,pKF->mImMask,5);
        if (isDilate)
        {
            cv::dilate(masktemp, pKF->mImMask, mKernel);
            if (pKF->mImMask.empty())
            {
                LOG(INFO) << "Dilate operation failed, pKF mask is empty";
                return; // masktemp;
            }
        }
        else
        {
            masktemp.copyTo(pKF->mImMask);
        }
        
        Config::GetInstance()->saveImage(pKF->mImMask > 0, "mask",  to_string(pKF->mnId)+"_ori_mask.png");
        // cv::Mat imMaskOld = cv::Mat::zeros(pKF->mImMask.size(),CV_8UC1);
        if(pKF->mnObjectOld>0){
            int nObjectOld = pKF->mnObjectOld;
            // vector<int> rotHist = vector<int>(30,0);
            // const float factor = 30 / 360.0f;
            vector<bool> check = vector<bool>(nObjectOld, false);
            for (int i = 1; i <= curObject; i++)
            {
                for (int j = 1; j <= nObjectOld; j++)
                {
                    if (check[j - 1])
                        continue;
                    float iou = cv::countNonZero((pKF->mImObject == i) & (pKF->mImObjectOld == j));
                    float all_ = cv::countNonZero((pKF->mImObjectOld  == j));
                    if (all_ == 0 || iou / all_ > 0.3)
                    {

                        // pKF->mImMask|=pKF->mImMaskOld;
                        check[j - 1] = true;
                        break;
                    }
                }
            }
            cv::Mat temp_cp = cv::Mat::zeros(masktemp.size(), CV_8UC1);
            cv::Mat temp_ob = cv::Mat::zeros(masktemp.size(), CV_8UC1);
            int tempnum = curObject;
            for (int i = 1; i <= nObjectOld; i++)
            {
                if (!check[i - 1])
                {
                    tempnum++;
                    cv::Mat temp_old = cv::Mat::ones(masktemp.size(), CV_8UC1) * tempnum;
                    copyTo(temp_old, temp_ob, pKF->mImObjectOld == i);
                }
            }
            pKF->mObjectNum = tempnum;
            cv::Mat kernel_sm = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
            cv::dilate(temp_ob,temp_cp,mKernel);
            cv::dilate(temp_ob, temp_ob, kernel_sm);
            temp_cp.copyTo(pKF->mImMask, temp_cp != 0);
            temp_ob.copyTo(pKF->mImObject, temp_ob != 0);
        }
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
        cv::erode(pKF->mImObject, pKF->mImObject, kernel);
        auto end = chrono::steady_clock::now();
        // saveMat(pKF->mImObject,"objectmask");
        // time consuming

        Config::GetInstance()->saveImage(pKF->mImMask > 0, "mask", to_string(pKF->mnId)+"_fuse_mask.png");
        Config::GetInstance()->saveImage(pKF->mImRGB, "mask",to_string(pKF->mnId)+"rgb.png");
        
        chrono::duration<double> diff = end - start;
        LOG(INFO) << "Time to update moving probability:" << setw(3) << diff.count() * 1000 << " ms";
        mvTimeMaskGeneration.push_back(diff.count());
        // return backMask;
    }
    
    void Semantic::GenerateMask(KeyFrame *pKF,cv::Mat &imFlow,vector<cv::Mat> vLabel, vector<cv::Mat> vObject, vector<int> vClassId, vector<float> &mDepth, bool isDilate)
    {
        LOG(INFO) << "Generate Mask for Frame: " << pKF->mnId;
        if (pKF->mObjectNum == 0)
            return; // cv::Mat::zeros(pKF->mImGray.size(),CV_8UC1);
        int curObject = pKF->mObjectNum;
        auto start = chrono::steady_clock::now();

        cv::Mat mask = cv::Mat::zeros(pKF->mImGray.size(), CV_8UC1);
        cv::Mat masktemp = cv::Mat::zeros(pKF->mImGray.size(), CV_8UC1);
        map<string, int>::iterator iter;
        vector<double> meanDepth;
        meanDepth.resize(curObject, 0);
        if (pKF->mImDepth.empty())
        {
            vector<int> cntKP(curObject, 0);
            for (int i = 0; i < pKF->N; i++)
            {
                if (pKF->mvDepth[i] < 0)
                    continue;
                if(pKF->mvKeys[i].pt.y>=pKF->mImGray.rows||pKF->mvKeys[i].pt.y<=0)
                    continue;
                if(pKF->mvKeys[i].pt.x>=pKF->mImGray.cols||pKF->mvKeys[i].pt.x<=0)
                    continue;
                for (int j = 0; j < curObject; j++)
                {
                    if (vObject[j].at<uchar>(pKF->mvKeys[i].pt.y, pKF->mvKeys[i].pt.x) != 0)
                    {
                        meanDepth[j] += pKF->mvDepth[i];
                        cntKP[j]++;
                    }
                }
            }
            for (int i = 0; i < curObject; i++)
            {
                meanDepth[i] /= cntKP[i];
            }
        }
        else
        {
            // map<int,int> label2id;
            for (int i = 0; i < curObject; i++)
            {
                cv::Mat depthMask = ((pKF->mImDepth > 0) & (vObject[i] != 0));
                cv::Mat curDepth;
                pKF->mImDepth.copyTo(curDepth, depthMask);
                meanDepth[i] = sum(curDepth)[0] / sum(depthMask / 255)[0];
                // label2id[vClassId[i]]=i+1;
            }
        }
        {

            pKF->mImLabel = vLabel[0];
            pKF->mImObject = vObject[0];

            // saveMat(vObject[0],"objectmask");
            for (int i = 1; i < curObject; i++)
            {
                cv::Mat iou = ((pKF->mImLabel != 0) & (vLabel[i] != 0));
                if (cv::sum(iou)[0] != 0)
                {
                    for (int r = 0; r < pKF->mImLabel.rows; r++)
                    {
                        for (int c = 0; c < pKF->mImLabel.cols; c++)
                        {
                            if (iou.at<uchar>(r, c) != 0)
                            {
                                int preObjectID = pKF->mImObject.at<uchar>(r, c);
                                if (meanDepth[preObjectID] > meanDepth[i])
                                {
                                    pKF->mImObject.at<uchar>(r, c) = (i + 1);
                                    pKF->mImLabel.at<uchar>(r, c) = vClassId[i];
                                }
                            }
                        }
                    }
                }
                vLabel[i].copyTo(vLabel[i], (pKF->mImLabel == 0));
                vObject[i].copyTo(vObject[i], (pKF->mImObject == 0));
                pKF->mImLabel += vLabel[i];
                pKF->mImObject += vObject[i];
            }
        }

        mask = pKF->mImLabel.clone();
        mask = (mask > 0);

        pKF->mImLabel.copyTo(masktemp, mask);
        // dilate masks to filter out features on edge
        if (isDilate)
        {
            cv::dilate(masktemp, pKF->mImLabel, mKernel);
            if (pKF->mImLabel.empty())
            {
                LOG(INFO) << "Dilate operation failed, pKF mask is empty";
                return; // masktemp;
            }
        }
        else
        {
            masktemp.copyTo(pKF->mImLabel);
        }
        // mask.copyTo(pKF->mImMaskOld);
        masktemp = cv::Mat::zeros(pKF->mImLabel.size(), CV_8UC1);
        pKF->mImObject.copyTo(masktemp, mask);

        pKF->mImMask = pKF->mImObject.clone();
        // cv::dilate(backMask,pKF->mImMask,5);
        if (isDilate)
        {
            cv::dilate(masktemp, pKF->mImMask, mKernel);
            if (pKF->mImMask.empty())
            {
                LOG(INFO) << "Dilate operation failed, pKF mask is empty";
                return; // masktemp;
            }
        }
        else
        {
            masktemp.copyTo(pKF->mImMask);
        }
        
        Config::GetInstance()->saveImage(pKF->mImMask > 0, "mask",  to_string(pKF->mnId)+"_ori_mask.png");
        // cv::Mat imMaskOld = cv::Mat::zeros(pKF->mImMask.size(),CV_8UC1);
        cv::Mat imMaskOld = cv::Mat::zeros(pKF->mImMask.size(),CV_8UC1);
        if(mnObjectPre>0){
            int nObjectOld = mnObjectPre;
            for (int j = 0; j < imMaskOld.rows; j++)
            {
                for (int k = 0; k < imMaskOld.cols; k++)
                {
                    int objectid=mImObjectPre.at<uchar>(j,k);
                    const int flow_x = imFlow.at<cv::Vec2f>(j,k)[0];
                    const int flow_y = imFlow.at<cv::Vec2f>(j,k)[1];
                    if (objectid!=0)
                    {
                        if(k+flow_x < imMaskOld.cols && k+flow_x > 0 && j+flow_y < imMaskOld.rows && j+flow_y > 0){
                            imMaskOld.at<uchar>(j+flow_y,k+flow_x) = objectid;
                        }
                    }
                }
            }
            vector<bool> check = vector<bool>(nObjectOld, false);
            for (int i = 1; i <= curObject; i++)
            {
                for (int j = 1; j <= nObjectOld; j++)
                {
                    if (check[j - 1])
                        continue;
                    float iou = cv::countNonZero((pKF->mImObject == i) & (imMaskOld == j));
                    float all_ = cv::countNonZero((imMaskOld == j));
                    if (all_ == 0 || iou / all_ > 0.3)
                    {
                        check[j - 1] = true;
                        break;
                    }
                }
            }
            cv::Mat temp_cp = cv::Mat::zeros(masktemp.size(), CV_8UC1);
            cv::Mat temp_ob = cv::Mat::zeros(masktemp.size(), CV_8UC1);
            int tempnum = curObject;
            for (int i = 1; i <= nObjectOld; i++)
            {
                if (!check[i - 1])
                {
                    tempnum++;
                    cv::Mat temp_old = cv::Mat::ones(masktemp.size(), CV_8UC1) * tempnum;
                    copyTo(temp_old, temp_ob, imMaskOld == i);
                }
            }
            pKF->mObjectNum = tempnum;
            cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
            cv::morphologyEx(temp_ob,temp_ob,cv::MORPH_CLOSE,kernel_open);
            // cv::dilate(temp_ob, temp_ob, kernel_open);
            cv::dilate(temp_ob,temp_cp,mKernel);
            temp_cp.copyTo(pKF->mImMask, temp_cp != 0);
            temp_ob.copyTo(pKF->mImObject, temp_ob != 0);
            cv::erode(pKF->mImObject, pKF->mImObject, kernel_open);
            // cv::imwrite("/out/debug/maskpre.png",pKF->mImMaskOld>0);
        }
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
        cv::erode(pKF->mImObject, pKF->mImObject, kernel);
        auto end = chrono::steady_clock::now();
        // saveMat(pKF->mImObject,"objectmask");
        // time consuming

        Config::GetInstance()->saveImage(pKF->mImMask > 0, "mask", to_string(pKF->mnId)+"_fuse_mask.png");
        Config::GetInstance()->saveImage(pKF->mImRGB, "mask",to_string(pKF->mnId)+"rgb.png");
        
        chrono::duration<double> diff = end - start;
        LOG(INFO) << "Time to update moving probability:" << setw(3) << diff.count() * 1000 << " ms";
        mvTimeMaskGeneration.push_back(diff.count());
        // return backMask;
    }
    
    void Semantic::GetObjectOrder(KeyFrame *pKF, vector<int> &order)
    {
        cv::Mat mask = pKF->mImObject;
        map<int, int>::iterator it;
        for (it = pKF->mObjectIdMap.begin(); it != pKF->mObjectIdMap.end(); it++)
        {
            int object = it->first;
            int boxMinX = pKF->mvBbox[object][0];
            int boxMinY = pKF->mvBbox[object][1];
            int boxMaxX = pKF->mvBbox[object][2];
            int boxMaxY = pKF->mvBbox[object][3];
            order.push_back(object);
            for (int r = boxMinX; r < boxMaxX; r++)
            {
                for (int c = boxMinY; c < boxMaxY; c++)
                {
                    int iobject = mask.at<uchar>(r, c);
                    if (iobject == 0)
                        continue;
                    else if (pKF->mObjectIdMap.count(iobject) == 0)
                        continue;
                    else if (iobject != object)
                    {
                        vector<int>::iterator itv = find(order.begin(), order.end(), iobject);
                        if (itv != order.end())
                            continue;
                        itv = find(order.begin(), order.end(), object);
                        order.insert(itv, iobject);
                    }
                }
            }
        }
    }

    void Semantic::SemanticThreadRe()
    {
        Frame *pKF, *frontKF, *backKF;
        vector<cv::Mat> vmLabel;
        vector<cv::Mat> vmObject;
        vector<cv::Mat> vmRequest;
        vector<Frame *> vKFs;
        vector<int> vBbox;
        vector<int> num;
        vector<int> vmClassId;
        size_t frontID, backID;

        vmLabel.reserve(mnBatchSize);
        vKFs.reserve(mnBatchSize);
        vmRequest.reserve(mnBatchSize);
        vmObject.reserve(mnBatchSize);
        num.reserve(mnBatchSize);

        // set cnn method
        LOG(INFO) << "Start Semantic  thread";
        mpSemanticCNN = make_shared<semantic_cnn::SemanticClient>(string("/semantic_server"), true);
        // this_thread::sleep_for(chrono::milliseconds(100));
        while (true)
        {
            if (!CheckNewFrames())
            {
                if (!CheckFinish())
                {
                    this_thread::sleep_for(chrono::milliseconds(1));
                    continue;
                }
                else
                {
                    LOG(INFO) << "Semantic thread Stopping ....";
                    break;
                }
            }

            vmRequest.clear();
            vKFs.clear();
            vmLabel.clear();
            vmObject.clear();
            vBbox.clear();
            num.clear();
            vmClassId.clear();
            unique_lock<mutex> lock(mMutexNewFrames);
            for (std::list<Frame *>::reverse_iterator ir = mlNewFrames.rbegin(); ir != mlNewFrames.rend(); ir++)
            {
                backKF = *ir;
                backID = backKF->mnId;
                // if(backID>mnLatestSemanticKeyFrameID){
                vKFs.push_back(backKF);
                vmRequest.push_back(backKF->mImRGB);
                if (vKFs.size() == mnBatchSize)
                {
                    break;
                }
                // }
            }
            // LOG(INFO) << "Back ID: " << backID;
            // LOG(INFO) << "vKFS queue size: " << vKFs.size();
            // the minimum request size is mBatchSize
            if (vKFs.size() < mnBatchSize)
            {
                lock.unlock();
                vKFs.clear();
                vmRequest.clear();
                // LOG(WARNING) << "KeyFrame size is less than batch size";
                continue;
            }

            mlNewFrames.clear();
            // key frame queue
            lock.unlock();

            LOG(INFO) << "=================================================";
            std::cout << "=================================================" << endl;
            mCurGray = vKFs[0]->mImGray;
            opticalFlowThread = async(launch::async, [this] {
                if(mLastGray.empty()) 
                    return cv::Mat();
                cv::Mat imFlow;
                cv::calcOpticalFlowFarneback(mLastGray,mCurGray,imFlow,0.5, 3, 15, 3, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN);
                return imFlow;
            });
            // ask for semantic result
            mpSemanticCNN->Semantic(vmRequest, vmLabel, num, vmObject, vBbox, vmClassId);

            // this_thread::sleep_for(chrono::milliseconds(30));
            // save semantic result and generate mask image
            if (num.size() != mnBatchSize)
            {
                LOG(ERROR) << "size of label is wrong";
            }
            int boxidx = 0, imgidx = 0;
            int j = 0;
            cv::Mat imFlow = opticalFlowThread.get();
            for (size_t i = 0; i < mnBatchSize; i++)
            {
                vector<cv::Mat> vObjectMask;
                vector<cv::Mat> vLabelMask;
                vector<int> vClassId;
                vector<float> meanDepth;
                if(num[i]==0) continue;
                vKFs[i]->mnObject = num[i];
                cv::Mat temp_img = vmRequest[i].clone();
                for (int k = 0; k < num[i]; k++, j++)
                {
                    // vector<int> temp;
                    cv::Rect temp;
                    temp.x = vBbox[j * 4];     //((box[0] + box[2]) / 2.);
                    temp.y = vBbox[j * 4 + 1]; //((box[1] + box[3]) / 2.);
                    temp.width = vBbox[j * 4 + 2] - vBbox[j * 4 + 0];
                    temp.height = vBbox[j * 4 + 3] - vBbox[j * 4 + 1];
                    vKFs[i]->mBbox.push_back(temp);
                    vClassId.push_back(vmClassId[imgidx]);
                    vObjectMask.push_back(vmObject[imgidx]);
                    vLabelMask.push_back(vmLabel[imgidx++]);
                }
            
                this->GenerateMask(vKFs[i], imFlow,vLabelMask, vObjectMask, vClassId, meanDepth, true);
                // vector<int> order;
                // this->GetObjectOrder(vKFs[i],order);
                vKFs[i]->UpdatePrioriMovingProbability();
                mpTracker->SetSemanticMask(vKFs[i]);
                // vKFs[i]->InformSemanticReady(true);
                mpAtlas->useKF = true;
                mnLatestSemanticKeyFrameID = vKFs[i]->mnId;
                mImObjectPre = vKFs[i]->mImObject.clone();
                mnObjectPre = vKFs[i]->mnObject;
                mLastGray = mCurGray.clone();
                // AddSemanticTrackRequest(vKFs[i]);
            }
        }
    }
    void Semantic::SemanticThreadReKey()
    {
        KeyFrame *pKF, *frontKF, *backKF;
        vector<cv::Mat> vmLabel;
        vector<cv::Mat> vmObject;
        vector<cv::Mat> vmRequest;
        vector<KeyFrame *> vKFs;
        vector<int> vBbox;
        vector<int> num;
        vector<int> vmClassId;
        size_t frontID, backID;

        vmLabel.reserve(mnBatchSize);
        vKFs.reserve(mnBatchSize);
        vmRequest.reserve(mnBatchSize);
        vmObject.reserve(mnBatchSize);
        num.reserve(mnBatchSize);

        // set cnn method
        LOG(INFO) << "Start Semantic  thread";
        mpSemanticCNN = make_shared<semantic_cnn::SemanticClient>(string("/semantic_server"), true);
        // this_thread::sleep_for(chrono::milliseconds(100));
        while (true)
        {
            if (!CheckNewKeyFrames())
            {
                if (!CheckFinish())
                {
                    this_thread::sleep_for(chrono::milliseconds(1));
                    continue;
                }
                else
                {
                    LOG(INFO) << "Semantic thread Stopping ....";
                    break;
                }
            }

            vmRequest.clear();
            vKFs.clear();
            vmLabel.clear();
            vmObject.clear();
            vBbox.clear();
            num.clear();
            vmClassId.clear();
            unique_lock<mutex> lock(mMutexNewKeyFrames);
            for (std::list<KeyFrame *>::reverse_iterator ir = mlNewKeyFrames.rbegin(); ir != mlNewKeyFrames.rend(); ir++)
            {
                backKF = *ir;
                backID = backKF->mnId;
                // if(backID>mnLatestSemanticKeyFrameID){
                vKFs.push_back(backKF);
                vmRequest.push_back(backKF->mImRGB);
                if (vKFs.size() == mnBatchSize)
                {
                    break;
                }
                // }
            }
            // LOG(INFO) << "Back ID: " << backID;
            // LOG(INFO) << "vKFS queue size: " << vKFs.size();
            // the minimum request size is mBatchSize
            if (vKFs.size() < mnBatchSize)
            {
                lock.unlock();
                vKFs.clear();
                vmRequest.clear();
                // LOG(WARNING) << "KeyFrame size is less than batch size";
                continue;
            }

            mlNewKeyFrames.clear();
            // key frame queue
            lock.unlock();

            LOG(INFO) << "=================================================";
            std::cout << "=================================================" << endl;
            mCurGray = vKFs[0]->mImGray;
            opticalFlowThread = async(launch::async, [this] {
                if(mLastGray.empty()) 
                    return cv::Mat();
                cv::Mat imFlow;
                cv::calcOpticalFlowFarneback(mLastGray,mCurGray,imFlow,0.5, 3, 15, 3, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN);
                return imFlow;
            });
            // ask for semantic result
            mpSemanticCNN->Semantic(vmRequest, vmLabel, num, vmObject, vBbox, vmClassId);

            // this_thread::sleep_for(chrono::milliseconds(30));
            // save semantic result and generate mask image
            if (num.size() != mnBatchSize)
            {
                LOG(ERROR) << "size of label is wrong";
            }
            int boxidx = 0, imgidx = 0;
            int j = 0;
            cv::Mat imFlow = opticalFlowThread.get();
            for (size_t i = 0; i < mnBatchSize; i++)
            {
                vector<cv::Mat> vObjectMask;
                vector<cv::Mat> vLabelMask;
                vector<int> vClassId;
                vector<float> meanDepth;
                if(num[i]==0) continue;
                vKFs[i]->mObjectNum = num[i];
                cv::Mat temp_img = vmRequest[i].clone();
                for (int k = 0; k < num[i]; k++, j++)
                {
                    // vector<int> temp;
                    // cv::Rect temp;
                    // temp.x = vBbox[j * 4];     //((box[0] + box[2]) / 2.);
                    // temp.y = vBbox[j * 4 + 1]; //((box[1] + box[3]) / 2.);
                    // temp.width = vBbox[j * 4 + 2] - vBbox[j * 4 + 0];
                    // temp.height = vBbox[j * 4 + 3] - vBbox[j * 4 + 1];
                    // vKFs[i]->mBbox.push_back(temp);
                    vClassId.push_back(vmClassId[imgidx]);
                    vObjectMask.push_back(vmObject[imgidx]);
                    vLabelMask.push_back(vmLabel[imgidx++]);
                }
                // this->GenerateMask(vKFs[i], vLabelMask, vObjectMask, vClassId, meanDepth, true);

                this->GenerateMask(vKFs[i], imFlow,vLabelMask, vObjectMask, vClassId, meanDepth, true);
                // vector<int> order;
                // this->GetObjectOrder(vKFs[i],order);
                vKFs[i]->UpdatePrioriMovingProbability();
                mpTracker->SetSemanticMask(vKFs[i]);
                // vKFs[i]->InformSemanticReady(true);
                mpAtlas->useKF = true;
                mnLatestSemanticKeyFrameID = vKFs[i]->mnId;
                mpLatestKF = vKFs[i];
                mImObjectPre = vKFs[i]->mImObject.clone();
                mnObjectPre = vKFs[i]->mObjectNum;
                Config::GetInstance()->saveImage(mImObjectPre > 0, "mask", to_string(vKFs[i]->mnId)+"_objectpre.png");
                mLastGray = mCurGray.clone();
                // AddSemanticTrackRequest(vKFs[i]);
            }
        }
    }
}