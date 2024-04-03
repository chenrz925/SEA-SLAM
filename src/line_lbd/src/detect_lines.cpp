/*
 * line_detection interface
 * Copyright Shichao Yang,2018, Carnegie Mellon University
 * Email: shichaoy@andrew.cmu.edu
 *
 */

#include <line_lbd/line_descriptor.hpp>

// #include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <ctime>
#include <line_lbd/line_lbd_allclass.h>

#include <ros/ros.h>

using namespace cv;
using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps,vector<string>& vimage_name)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
        vimage_name.push_back(ss.str());
    }
}

int main( int argc, char** argv )
{
      /* get parameters from comand line */
      if(argc<2)
      {
          std::cout<<"Provide an image name"<<endl;
          return -1;
      }

      // STEP 【1.初始化，读取图像并定义变量.】
      // 初始化节点与命名空间.
      ros::init(argc, argv, "detect lines");
      ros::NodeHandle nh; 

      // 读取图像 raw_img.
      std::string images_folder(argv[1]);


      // 定义变量，使用算法，是否保存线段信息.
      std::string save_folder;
      bool use_LSD_algorithm;
      bool save_to_imgs;
      bool save_to_txts;
    //   nh.param<std::string>("images_folder", save_folder);
      nh.param<std::string>("save_folder", save_folder, "$(find line_lbd)/data");
      nh.param<bool>("use_LSD_algorithm",use_LSD_algorithm,true);
      nh.param<bool>("save_to_imgs",save_to_imgs,true);
      nh.param<bool>("save_to_txts",save_to_txts,true);
      
      // STEP 【2.设置检测类.】
      int numOfOctave_ = 1;
      float Octave_ratio = 2.0;  
      // line_lbd_detect：线检测类，定义一个线检测类的对象 line_lbd_ptr.
      line_lbd_detect* line_lbd_ptr = new line_lbd_detect(numOfOctave_,Octave_ratio); 
      // 使用 LSD 算法.
      line_lbd_ptr->use_LSD = true;
      //设置长度阈值维 15.
      line_lbd_ptr->line_length_thres = 5;  // remove short edges

        vector<string> vstrImageLeft;
        vector<string> vstrImageRight;
        vector<string> vstrImageName;
        vector<double> vTimestamps;

        LoadImages(images_folder, vstrImageLeft, vstrImageRight, vTimestamps,vstrImageName);
        int nImages = vstrImageLeft.size();
        for (int i = 0; i < nImages; i++)
        {
            cv::Mat raw_img = cv::imread(vstrImageLeft[i],cv::IMREAD_UNCHANGED);
        
            // STEP 【3.线段检测 detect_raw_lines.】
            // 使用作者的线检测器可以在 detect_raw_lines() 函数中选择使用 LSD 还是 edline.
            cv::Mat out_edges;
            std::vector< KeyLine> keylines_raw,keylines_out;
            // NOTE 线检测 输入原始图像：raw_img  输出检测线段：keylines_raw.
            line_lbd_ptr->detect_raw_lines(raw_img,keylines_raw);
            // 剔除过短的线段，最后保存为 keylines_out.
            line_lbd_ptr->filter_lines(keylines_raw,keylines_out);
            
            // STEP 【4.图像显示，线段绘制】
            if( raw_img.channels() == 1 )
                cvtColor( raw_img, raw_img, COLOR_GRAY2BGR );
            cv::Mat raw_img_cp;
            // NOTE 绘制线段 keylines_out 到 raw_img_cp 图像中. 
            cout<<vstrImageLeft[i]<<"detect keylines: "<<keylines_out.size()<<endl;
            //   drawKeylines(raw_img, keylines_out, raw_img_cp, cv::Scalar( 0, 150, 0 ),2); // B G R
            //   imshow( "Line detector", raw_img_cp );
            //   waitKey(0);
            
            // STEP 【5.信息保存】
            // 保存包含线段的图像.
            if (save_to_imgs)
            {
                std::string img_save_name = save_folder+"saved_edges.jpg";
                cv::imwrite(img_save_name,raw_img_cp);
            }
            // 保存所有线段的坐标信息.
            if (save_to_txts)
            {
                std::string txt_save_name = images_folder+"/saved_edges/"+vstrImageName[i]+".txt";
                ofstream resultsFile;
                resultsFile.open(txt_save_name);
                for (int j=0; j<keylines_out.size(); j++)
                {
                    resultsFile <<keylines_out[j].startPointX <<"\t" <<keylines_out[j].startPointY  <<"\t"
                                <<keylines_out[j].endPointX   <<"\t" <<keylines_out[j].endPointY    <<endl;
                }
                resultsFile.close();
            }
        }
}
