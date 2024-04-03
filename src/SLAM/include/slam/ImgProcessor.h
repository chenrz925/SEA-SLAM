#include <opencv2/highgui/highgui.hpp>
using namespace cv;
int GetMatMidVal(cv::Mat& img);
void GetMatMinMaxThreshold(cv::Mat& img, int& minval, int& maxval, float sigma);
cv::Mat GetCannyRes(const cv::Mat &src);
cv::Mat GetWateredRes(cv::Mat &src);
void RegionGrow(cv::Mat &src, cv::Point2i pt,cv::Mat &depth, cv::Mat &mask,int th);
void  RegionGrowStatic(cv::Mat &src, cv::Point2i pt, cv::Mat &depth,cv::Mat &mask,int th);
void RegionGrowWithWatered( cv::Mat &src,  cv::Point2i pt,  cv::Mat &depth, cv::Mat &mask,cv::Mat &watered,int th);
void RegionGrow(Mat &src, Point2i pt, Mat &mask,Mat &vis);
void RegionGrowing(cv::Mat &im,cv::Mat & depth,cv::Mat & gray,int x,int y,const float &dThresh,float &gThresh);
void lockhole(cv::Mat &img);

#define DEFAULT_M 20
#define USE_DEFAULT_S -1

class Superpixels{

public:
    Superpixels(Mat& img, float m = DEFAULT_M, float S = USE_DEFAULT_S); // calculates the superpixel boundaries on construction
    
    Mat viewSuperpixels(); // returns image displaying superpixel boundaries
    Mat colorSuperpixels(); // recolors image with average color in each cluster
    // map<int, Point>  getCentersMap(); // returns the labels and their cluster centers
    std::vector<Point> getCenters(); // centers indexed by label
    Mat getLabels(); // per pixel label
    
protected:
    Mat img; // original image
    Mat img_f; // scaled to [0,1]
    Mat img_lab; // converted to LAB colorspace

    // used to store the calculated results
    Mat show;
    Mat labels; 
    
    float m; // compactness parameter
    float S; // window size

    int nx, ny; // cols and rows
    float dx, dy; // steps
    
    std::vector<Point> centers; // superpixel centers
    
    void calculateSuperpixels();
    float dist(Point p1, Point p2); // 5-D distance between pixels in LAB space

    const static Mat sobel;
};
class lime
{  
public:
     int channel;
     cv::Mat out_lime;


public:
    lime(){};
    cv::Mat lime_enhance(cv::Mat& src);

    static inline float compare(float& a,float& b,float& c)
    {
        return fmax(a,fmax(b,c));
    }
    void Illumination(cv::Mat& src,cv::Mat& out);

    void Illumination_filter(cv::Mat& img_in,cv::Mat& img_out);

};