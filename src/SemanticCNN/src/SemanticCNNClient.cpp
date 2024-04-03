#include "SemanticCNNClient.h"
using namespace std;
namespace semantic_cnn
{
    SemanticClient::SemanticClient(const string &strName, const bool bSpin) : cActionClient_(strName, bSpin)
    {
        ROS_INFO("Waiting for action server to start.");
        cout << "Waiting for action server to start." << endl;
        cActionClient_.waitForServer();
        ROS_INFO("Action server ready.");
        liFactoryId_ = -1;
    }
    void SemanticClient::Semantic(const vector<cv::Mat> &vmatImage, vector<cv::Mat> &vmatOutLabel,vector<int> &ObjectNum,
                        vector<cv::Mat> &OutObject,vector<int> &OutBbox,vector<int> &OutClassId)
    {
        //set time out second
        double time_out_sec = 30.0;
        // prepare and sent goal to server
        cout << "------------------------------------------" << endl;
        int imagev_size = vmatImage.size();
        if (imagev_size == 0)
        {
            ROS_ERROR( "Image vector empty.");
            cout << "Image vector empty." << endl;

            return;
        }
        semanticGoal goal;
        goal.id = ++liFactoryId_;
        for (size_t i = 0; i < imagev_size; i++)
        {
            cv_bridge::CvImage cvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, vmatImage[i]);
            goal.image.push_back(*(cvImage.toImageMsg()));
        }
        cout << "Sending request:" << goal.id << endl;
        ROS_INFO("Sending request: %d" , goal.id);
        cActionClient_.sendGoal(goal);

        //wait for server return
        bool finished_before_timeout = cActionClient_.waitForResult(ros::Duration(time_out_sec)); //set time out 30sec
        if (finished_before_timeout)
        {
            actionlib::SimpleClientGoalState state = cActionClient_.getState();
            ROS_INFO("Action finished: %s", state.toString().c_str());
        }
        else
        {
            ROS_INFO("Action did not finish before the time out %d" , time_out_sec);
        }
        mResult = cActionClient_.getResult();
        int idx = 0,imgidx=0;
        for (size_t i = 0; i < imagev_size; i++)
        {

            int N = mResult->object_num[i];
            ObjectNum.push_back(N);
            for(int j=0;j<N;j++)
            {
                cv::Mat label = cv_bridge::toCvCopy(mResult->label[imgidx])->image;
                vmatOutLabel.push_back(label);
                label = cv_bridge::toCvCopy(mResult->label_object[imgidx])->image;
                OutObject.push_back(label);
                OutClassId.push_back(mResult->class_id[imgidx]);
                imgidx++;
            }
            for(int j = 0;j<N*4;j++)
            {
                OutBbox.push_back(mResult->bbox[idx]);
                idx++;
            }
        }
    }

    void SemanticClient::RequestSemantic(const vector<cv::Mat>& vmatImages)
    {
            //set time out second
        double time_out_sec = 30.0;
        // prepare and sent goal to server
        cout << "------------------------------------------" << endl;
        int imagev_size = vmatImages.size();
        if (imagev_size == 0)
        {
           ROS_ERROR("Image vector empty.");
            cout << "Image vector empty." << endl;
            return;
        }
        semanticGoal goal;
        goal.id = ++liFactoryId_;
        for (size_t i = 0; i < imagev_size; i++)
        {
            cv_bridge::CvImage cvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, vmatImages[i]);
            goal.image.push_back(*(cvImage.toImageMsg()));
        }
        cout << "Sending request:" << goal.id << endl;
        ROS_INFO("Sending request: %d" ,goal.id);
        cActionClient_.sendGoal(goal,boost::bind(&SemanticClient::doneCallBack,this,_1,_2),
                                    Client::SimpleActiveCallback(),
                                    Client::SimpleFeedbackCallback());

        //wait for server return
        bool finished_before_timeout = cActionClient_.waitForResult(ros::Duration(time_out_sec)); //set time out 30sec
        if (finished_before_timeout)
        {
            actionlib::SimpleClientGoalState state = cActionClient_.getState();
            ROS_INFO("Action finished: %s" , state.toString().c_str());
        }
        else
        {
            ROS_INFO("Action did not finish before the time out %f",time_out_sec);
        }
        // mResult = cActionClient_.geResult();
        // for (size_t i = 0; i < imagev_size; i++)
        // {
        //     cv::Mat label = cv_bridge::toCvCopy(mResult->label[i]->image);
        //     vmatOutLabel.push_back(label);
        // }
    }

    void SemanticClient::doneCallBack(const actionlib::SimpleClientGoalState &aState, const semanticResultConstPtr &seResult)
    {
        ROS_INFO("Finished in state [%s]",aState.toString().c_str());
        ROS_INFO("Answer: %d" , seResult->id);
    }
}