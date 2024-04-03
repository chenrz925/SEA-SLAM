/* 线检测接口.
 * line_detection interface
 * Copyright Shichao Yang,2016, Carnegie Mellon University
 * Email: shichaoy@andrew.cmu.edu
 *
 */

# pragma once
#include <line_lbd/line_descriptor/descriptor.hpp>

// #include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
using namespace Eigen;
// #define MATCHES_DIST_THRESHOLD 25  // this might need to change with image size


using namespace cv::line_descriptor;


// BRIEF 线检测类 【line_lbd_detect】
class line_lbd_detect
{
public:
  
    line_lbd_detect(int numoctaves=1, float octaveratio=1);     // TODO 这两个参数？
    int numoctaves_;
    float octaveratio_;
    
    // 是否使用LSD检测线.
    bool use_LSD=true;  // use LSD detector or edline detector    
	// Edline usually detects longer but fewer lines. LSD detects more, but may short, broken
    // 线阈值，过低会剔除.
    float line_length_thres=15;
    
    /* create a pointer to a BinaryDescriptor object with deafult parameters, edline */
    // 使用deafult参数创建指向BinaryDescriptor对象的指针，edline
    cv::Ptr<BinaryDescriptor> lbd;
  
    /* create a BinaryDescriptorMatcher object */
    cv::Ptr<BinaryDescriptorMatcher> bdm;
    
    /* create a pointer to LSD line detector */
    // 创建一个指向 LSD 检测器的指针.
    cv::Ptr<LSDDetector> lsd;
    
    // NOTE 线检测函数.
    void detect_raw_lines(const cv::Mat& gray_img, std::vector< KeyLine>& keylines_out);  // without line thresholding
    void detect_raw_lines(const cv::Mat& gray_img, std::vector<std::vector< KeyLine>>& keyline_octaves);  // without line thresholding, multi-octaves     
    void detect_raw_lines(const cv::Mat& gray_img, cv::Mat& linesmat_out, bool downsample_img=false);  /* output is n*4 32F mat. x1 y1 x2 y2 */    
    
    // 获取线描述.
    void get_line_descriptors(const cv::Mat& gray_img, const cv::Mat& linesmat_src, cv::Mat& line_descrips);

    /* filter edges with line thresholding, output is n*4 mat. x1 y1 x2 y2 */
    // NOTE 线段筛选函数，使用线阈值处理进行筛选.
    void filter_lines(std::vector< KeyLine>& keylines_in, std::vector< KeyLine>& keylines_out);
    void filter_lines(std::vector< KeyLine>& keylines_in, Eigen::MatrixXd& keylines_out);
    
    // detect raw edges then filter the length
    void detect_filter_lines(const cv::Mat& gray_img, cv::Mat& linesmat_out);
    void detect_filter_lines(const cv::Mat& gray_img, std::vector< KeyLine>& keylines_out);
    
    // NOTE 检测边缘病计算器描述符.
    /* detect edges and compute edge descriptors */
    void detect_descrip_lines(const cv::Mat& gray_img, cv::Mat& linesmat_out, cv::Mat& line_descrips);
    /* detect edges and compute edge descriptors */
    void detect_descrip_lines(const cv::Mat& gray_img, std::vector< KeyLine>& keylines_out, cv::Mat& line_descrips);
    
    // TODO each octaves would be a separate vector.
    void detect_descrip_lines_octaves(const cv::Mat& gray_img, std::vector<std::vector< KeyLine>>& keylines_out, std::vector<cv::Mat>& line_descrips);
    
    // NOTE 计算一组边缘描述子的最佳匹配.
    /* compute the best matches for a set of edge descriptors. for each query descriptors, find the closest descriptor in traings. the smaller threshold, the stricter match */
    void match_line_descrip(const cv::Mat& descrips_query, const cv::Mat& descrips_train, std::vector<cv::DMatch>& good_matches, float matching_dist_thres=25);
    
    /* compute the best matches for a set of edge descriptors. for each query descriptors, find the closest descriptor in traings */
//     void match_line_descrip_nearby(const cv::Mat& descrips_query, const cv::Mat& descrips_train, std::vector<cv::DMatch>& good_matches, float matching_dist_thres=25);
    
};

/* change vector of keylines to cvMat (32F float). */
// 向量格式转换.cvMat——vector< KeyLine>.
void keylines_to_mat(const std::vector< KeyLine>& keylines_src, 
                     cv::Mat& linesmat_out, 
                     float scale=1);
/* change cvMat (float) to vector of keylines */
void mat_to_keylines(const cv::Mat& linesmat_src, 
                     std::vector< KeyLine>& keylines_out, 
			         int raw_img_width, 
                     int raw_img_height, 
                     float raw_length_threshold=0, 
                     float close_boundary_threshold=0, 
                     float scaling=1,
			         int octave_id=0,
                     float each_octave_scale=1);