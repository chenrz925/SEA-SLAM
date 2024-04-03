#ifndef _SEMANTICCNN_H_
#define _SEMANTICCNN_H_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <actionlib/client/simple_action_client.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <semantic_cnn/semanticAction.h>
#include <glog/logging.h>

using namespace std;
namespace semantic_cnn
{
    typedef actionlib::SimpleActionClient<semantic_cnn::semanticAction> Client;
    class SemanticClient
    {
    public:
        typedef shared_ptr<SemanticClient> Ptr;
        semanticResultConstPtr mResult;
        SemanticClient(const string& strName, const bool bSpin);
        // void Semantic(const vector<cv::Mat> &vmatImage, vector<cv::Mat> &vmatOutLabel);
        void Semantic(const vector<cv::Mat> &vmatImage, vector<cv::Mat> &vmatOutLabel,vector<int> &ObjectNum,
                        vector<cv::Mat> &OutObject,vector<int> &OutBbox,vector<int> &OutClassId);

        void RequestSemantic(const std::vector<cv::Mat>& vmatImages);
        void doneCallBack(const actionlib::SimpleClientGoalState &aState, const semanticResultConstPtr &seResult);

    private:
        Client cActionClient_;
        long int liFactoryId_;
    };
}
#endif