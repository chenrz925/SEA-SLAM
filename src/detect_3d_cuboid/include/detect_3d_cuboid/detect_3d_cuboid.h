/** @file detect_3d_cuboid.h	定义立方体类
 * 
 */

#pragma once

// std c
#include <string>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include "detect_3d_cuboid/matrix_utils.h"
using namespace std;
// BRIEF matlab cuboid struct
class cuboid // matlab cuboid struct. cuboid on ground. only has yaw, no obj roll/pitch
{
public:  
    Eigen::Vector3d pos;      // 位置，xyz.
    Eigen::Vector3d scale;    // 尺度，长宽高.
    double rotY;              // 方向.
    
    Eigen::Vector2d box_config_type;            // configurations, vp1 left/right
    Eigen::Matrix2Xi box_corners_2d;            // 图像平面的 2D坐标 2*8
    Eigen::Matrix3Xd box_corners_3d_world;      // 世界坐标系下的 3D 坐标 3*8
    
    Eigen::Vector4d rect_detect_2d;             // 2D检测框.% 2D bounding box (might be expanded by me)
    double edge_distance_error;
    double edge_angle_error;
    double normalized_error;  // normalized distance+angle NOTE 
    double skew_ratio;                          // 歪斜比.
    double down_expand_height;
    double camera_roll_delta;
    double camera_pitch_delta;

    void print_cuboid();	// print pose information
};
typedef std::vector<cuboid*> ObjectSet;  // for each 2D box, the set of generated 3D cuboids
/*
printing cuboids info....
【pos】     -1.58339 0.373187 0.300602
【scale】   0.155737 0.436576 0.300602
【rotY】    -2.90009
【box_config_type】   1  1
【box_corners_2d】 
503 279 213 430 559 261 174 459
245 396 319 200  56 184 116  23
【box_corners_3d_world】 
-1.6302     -1.83902   -1.53659   -1.32776    -1.6302    -1.83902    -1.53659   -1.32776
-0.087966   0.759848    0.83434   -0.0134734  -0.087966   0.759848    0.83434   -0.0134734
	0          0          0          0       0.601204   0.601204    0.601204   0.601204
*/

// BRIEF cam_pose_infos：包括相机位姿信息的结构体.
struct cam_pose_infos
{
      Eigen::Matrix4d transToWolrd;       // 4*4的变换矩阵（位姿）.
      Eigen::Matrix3d Kalib;              // 3*3的相机内参.
      
      Eigen::Matrix3d rotationToWorld;    // 3*3的旋转矩阵.
      Eigen::Vector3d euler_angle;        // 3维向量的欧拉角，分别表示 roll，pitch和yaw角.
      Eigen::Matrix3d invR;               // 旋转矩阵的逆.
      Eigen::Matrix3d invK;               // 相机内参的逆矩阵.
      Eigen::Matrix<double, 3, 4> projectionMatrix;      // 投影矩阵.
      Eigen::Matrix3d KinvR;              // Kalib*invR
      double camera_yaw;
};

// BRIEF  detect_3d_cuboid 定义了一个立方体检测的类.
class detect_3d_cuboid
{
public:
      cam_pose_infos cam_pose;
      cam_pose_infos cam_pose_raw;
      detect_3d_cuboid(float fx,float fy,float cx,float cy);
      // 传递相机内参，相机变换矩阵.
      void set_calibration(const Eigen::Matrix3d& Kalib);
      void set_cam_pose(const Eigen::Matrix4d& transToWolrd);

      // object detector needs image, camera pose, and 2D bounding boxes(n*5, each row: xywh+prob)  long edges: n*4.  all number start from 0
      // NOTE 立方体物体检查函数 detect_cuboid().
      // 参数：图像 rgb_img，相机位姿 transToWolrd，2D边界框 obj_bbox_coors(n*5, 每行: xywh+prob)
      //      长边 edges n*4 ，均从 0 开始.
      // void detect_cuboid(const cv::Mat& rgb_img, 
      //                    const Eigen::Matrix4d& transToWolrd,
      //                    const std::vector<cv::Rect>& obj_bbox_coors,
      //                    std::vector<cv::line_descriptor::KeyLine>& all_lines_raw, 									
	// 		       std::vector<ObjectSet>& all_object_cuboids);     // TODO  all_object_cuboids 是什么？？
      void detect_cuboid(	const cv::Mat& rgb_img,
                              const Eigen::Matrix4d& transToWolrd, 
                              const std::vector<cv::Rect>& obj_bbox_coors,
                              Eigen::MatrixXd& all_lines_raw, 
                              std::vector<ObjectSet>& all_object_cuboids);
      ///@param   whether_plot_detail_images	是否显示细节图：边缘检测，Canny检测，距离归一化.
      bool whether_plot_detail_images = false;
      ///@param   whether_plot_final_images	是否显示原始图+边框.
      bool whether_plot_final_images = false;	// 显示检测结果图.
      bool whether_save_final_images = false; 	// 保存检测结果图.
      cv::Mat cuboids_2d_img;                   // 带有立方体提案的2D图像. opencv 格式.

      bool print_details = false;
      /** @param  print_details		是否输出检测的细节信息.
       * Configuration 2 fails at corner 4, outside box
       * Configuration 1 fails at edge 1-4, too short
       * Configuration 2 fails at corner 4, outside box
       */
      
      // 提案生成的重要参数.
      // important mode parameters for proposal generation.
      bool consider_config_1 = true;                  // false true
      bool consider_config_2 = true;                  // TODO 
      bool whether_sample_cam_roll_pitch = false;      // sample camera roll pitch in case don't have good camera pose TODO
      bool whether_sample_bbox_height = false;        // sample object height as raw detection might not be accurate TODO

      int max_cuboid_num = 1;  	                  // 最终返回的立方体的个数.
      double nominal_skew_ratio = 1;                  // normally this 1, unless there is priors TODO 为什么是1？？
      double max_cut_skew = 3;                        // TODO 最大歪斜比？
};