#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <queue>
#include "math.h"
#include "ImgProcessor.h"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;
  
 //求Mat的中位数
int GetMatMidVal(cv::Mat& img)
{
  //判断如果不是单通道直接返回128
  if (img.channels() > 1) return 128;
  int rows = img.rows;
  int cols = img.cols;
  //定义数组
  float mathists[256] = { 0 };
  //遍历计算0-255的个数
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      int val = img.at<uchar>(row, col);
      mathists[val]++;
    }
  }
 
 
  int calcval = rows * cols / 2;
  int tmpsum = 0;
  for (int i = 0; i < 255; ++i) {
    tmpsum += mathists[i];
    if (tmpsum > calcval) {
      return i;
    }
  }
  return 0;
}
//求自适应阈值的最小和最大值
void GetMatMinMaxThreshold(cv::Mat& img, int& minval, int& maxval, float sigma)
{
  int midval = GetMatMidVal(img);
//   cout << "midval:" << midval << endl;
  // 计算低阈值
  minval = saturate_cast<uchar>((1.0 - sigma) * midval);
  //计算高阈值
  maxval = saturate_cast<uchar>((1.0 + sigma) * midval);
}

cv::Mat GetCannyRes(const Mat& src)
{
        //转换灰度图
    cv::Mat gray;
    //获取自适应阈值
    int minthreshold, maxthreshold;
    // cv::cvtColor(gray,gray,)
    src.convertTo(gray,CV_8UC1);
    GetMatMinMaxThreshold(gray, minthreshold, maxthreshold,0.3);
    // cout << "min:" << minthreshold << "  max:" << maxthreshold << endl;
    //Canny边缘提取
    cv::Canny(gray, gray, minthreshold, maxthreshold);
    // cv::imwrite("/out/tum/fr3_walking_static_yolact/depth/1.png",gray);
    return gray;
}

Mat GetWateredRes(Mat& src)
{
	// src = GetCannyRes(src);
	vector<vector<Point>> contours;  
	vector<Vec4i> hierarchy; 
	findContours(src,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());  
	Mat imageContours=Mat::zeros(src.size(),CV_8UC1);  //轮廓	
	Mat marks(src.size(),CV_32S);   //Opencv分水岭第二个矩阵参数
	marks=Scalar::all(0);
	int index = 0;
	int compCount = 0;
	for( ; index >= 0; index = hierarchy[index][0], compCount++ ) 
	{
		//对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点
		drawContours(marks, contours, index, Scalar::all(compCount+1), 1, 8, hierarchy);
		// drawContours(imageContours,contours,index,Scalar(255),1,8,hierarchy);  
	}
	return marks;
}

void RegionGrow(Mat &src, Point2i pt, Mat &depth,Mat &mask,int th)
{
	Point2i ptGrowing;						//待生长点位置
	int nGrowLable = 0,nCurValue = 0;								//标记是否生长过
	int nSrcValue = src.at<uchar>(pt.y, pt.x);								//生长起点灰度值
  double initD = depth.at<float>(pt.y, pt.x);
  //生长方向顺序数据
	int DIR[8][2] = {  { 0, -1 }, { -1, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, -1 }, { -1, 1 },{ -1, -1 } };
	queue<Point2i> vcGrowPt;						//生长点栈
	vcGrowPt.push(pt);	
	int npoint=100;
  while (!vcGrowPt.empty()&&npoint)						//生长栈不为空则生长
	{
		pt = vcGrowPt.front();						//取出一个生长点
		vcGrowPt.pop();
    npoint--;
		//分别对八个方向上的点进行生长
		for (int i = 0; i<4; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;
      int d=depth.at<float>(ptGrowing.y, ptGrowing.x);
			nGrowLable = mask.at<uchar>(ptGrowing.y, ptGrowing.x);		//当前待生长点的灰度值
      nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
			if (nGrowLable!=255&&nCurValue!=nSrcValue)					//如果标记点还没有被生长
			{
        double residual = abs(d-initD);
				if (residual < th)					//在阈值范围内则生长
				{
					src.at<uchar>(ptGrowing.y, ptGrowing.x) = nSrcValue;		//标记为白色
					vcGrowPt.push(ptGrowing);					//将下一个生长点压入栈中
				}
			}
		}
	}
}

void  RegionGrowStatic(Mat &src, Point2i pt, Mat &depth,Mat &mask,int th)
{
Point2i ptGrowing;						//待生长点位置
	int nGrowLable = 0,nCurValue = 0;								//标记是否生长过
	int nSrcValue = src.at<uchar>(pt.y, pt.x);								//生长起点灰度值
  	double initD = depth.at<float>(pt.y, pt.x);  //生长方向顺序数据
	int DIR[8][2] = {  { 0, -1 }, { -1, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, -1 }, { -1, 1 },{ -1, -1 } };
	queue<Point2i> vcGrowPt;						//生长点栈
	vcGrowPt.push(pt);	
	int npoint=10000;
  while (!vcGrowPt.empty()&&npoint)						//生长栈不为空则生长
	{
		pt = vcGrowPt.front();						//取出一个生长点
		vcGrowPt.pop();
    npoint--;
		//分别对八个方向上的点进行生长
		for (int i = 0; i<4; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;
      int d=depth.at<float>(ptGrowing.y, ptGrowing.x);
			nGrowLable = mask.at<uchar>(ptGrowing.y, ptGrowing.x);		//当前待生长点的灰度值
      nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
			if (nGrowLable!=255&&nCurValue!=nSrcValue)					//如果标记点还没有被生长
			{
        double residual = abs(d-initD);
				if (residual < th)					//在阈值范围内则生长
				{
					src.at<uchar>(ptGrowing.y, ptGrowing.x) = nSrcValue;		//标记为白色
					vcGrowPt.push(ptGrowing);					//将下一个生长点压入栈中
				}
			}
		}
	}
}

void RegionGrowWithWatered( cv::Mat &src,  cv::Point2i pt,  cv::Mat &depth, cv::Mat &mask,cv::Mat &watered,int th)
{
	Point2i ptGrowing;						//待生长点位置
	int nGrowLable = 0,nCurValue = 0;								//标记是否生长过
	int nSrcValue = src.at<uchar>(pt.y, pt.x);								//生长起点灰度值
  double initD = depth.at<float>(pt.y, pt.x);  //生长方向顺序数据
	int waterMark=watered.at<uchar>(pt.y, pt.x);
	int DIR[8][2] = {  { 0, -1 }, { -1, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, -1 }, { -1, 1 },{ -1, -1 } };
	queue<Point2i> vcGrowPt;						//生长点栈
	vcGrowPt.push(pt);	
	int npoint=1000;
  while (!vcGrowPt.empty()&&npoint)						//生长栈不为空则生长
	{
		pt = vcGrowPt.front();						//取出一个生长点
		vcGrowPt.pop();
    npoint--;
		//分别对八个方向上的点进行生长
		for (int i = 0; i<4; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;
      		int d=depth.at<float>(ptGrowing.y, ptGrowing.x);
			nGrowLable = mask.at<uchar>(ptGrowing.y, ptGrowing.x);		//当前待生长点的灰度值
			nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
			int curMark = watered.at<uchar>(ptGrowing.y, ptGrowing.x);
			if (nGrowLable!=255&&nCurValue!=nSrcValue&&waterMark==curMark)					//如果标记点还没有被生长
			{
        		double residual = abs(d-initD);
				if (residual < th)					//在阈值范围内则生长
				{
					src.at<uchar>(ptGrowing.y, ptGrowing.x) = nSrcValue;		//标记为白色
					vcGrowPt.push(ptGrowing);					//将下一个生长点压入栈中
				}
			}
		}
	}
}

void RegionGrow(Mat &src, Point2i pt, Mat &mask,Mat &vis)
{
	Point2i ptGrowing;						
	int nGrowLable = 0,nCurValue = 0;								
	int nSrcValue = src.at<uchar>(pt.y, pt.x);	
	// if(nSrcValue!=0)
	// 	return ;	
	int DIR[8][2] = {  { 0, -1 }, { -1, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, -1 }, { -1, 1 },{ -1, -1 } };
	queue<Point2i> vcGrowPt;						//生长点栈
	Mat hassMarked = vis.clone();
	hassMarked.at<uchar>(pt.y,pt.x)=1;
	vcGrowPt.push(pt);	
	int npoint=1000;
  	while (!vcGrowPt.empty()&&npoint)						//生长栈不为空则生长
	{
		npoint--;
		pt = vcGrowPt.front();						//取出一个生长点
		vcGrowPt.pop();
		for (int i = 0; i<4; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (ptGrowing.x <= 0 || ptGrowing.y <= 0 || ptGrowing.x >=(src.cols - 1) || (ptGrowing.y >= src.rows - 1))
				continue;
			nGrowLable = mask.at<uchar>(ptGrowing.y, ptGrowing.x);		//当前待生长点的灰度值
      		nCurValue = (int)src.at<uchar>(ptGrowing.y, ptGrowing.x);
			if(nGrowLable!=0) continue;
			if((int)vis.at<uchar>(ptGrowing.y, ptGrowing.x)==1||hassMarked.at<uchar>(ptGrowing.y, ptGrowing.x)==1) continue;
			if(nSrcValue!=nCurValue)
			{
				vis.at<uchar>(ptGrowing.y, ptGrowing.x)=1;
			}
			src.at<uchar>(ptGrowing.y, ptGrowing.x) = nSrcValue;	
			hassMarked.at<uchar>(ptGrowing.y, ptGrowing.x)=1;
			vcGrowPt.push(ptGrowing);	
		}
	}
}

void lockhole(cv::Mat &img)
{		
	int DIR[8][2] = {  { 0, -1 }, { -1, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, -1 }, { -1, 1 },{ -1, -1 } };
  	for(int x=1;x<img.rows-1;x++)
	{
		for(int y=1;y<img.cols-1;y++)
		{
			int cnt=0;
			for(int i=0;i<4;i++){
				int nx = x + DIR[i][0];
				int ny = y + DIR[i][1];
				if(img.at<uchar>(nx,ny)==255)
					cnt++;
			}
			if(cnt>=2)
			{
				img.at<uchar>(x,y)==254;
			}
		}
	}
}
void RegionGrowing(cv::Mat &im,cv::Mat & depth,cv::Mat & gray,int x,int y,const float &dThresh,float &gThresh)
{
	cv::Mat J = cv::Mat::zeros(im.size(),CV_32F);
	int objectId = im.at<uchar>(y,x);
    float reg_mean = depth.at<float>(y,x);
    int reg_size = 1;
	cv::Mat show;
	im.copyTo(show);
	cvtColor(show,show,COLOR_GRAY2RGB);
    //Neighbor locations (footprint)
    cv::Mat neigb(4,2,CV_32F);
    neigb.at<float>(0,0) = -1;
    neigb.at<float>(0,1) = 0;
    neigb.at<float>(1,0) = 1;
    neigb.at<float>(1,1) = 0;
    neigb.at<float>(2,0) = 0;
    neigb.at<float>(2,1) = -1;
    neigb.at<float>(3,0) = 0;
    neigb.at<float>(3,1) = 1;
	queue<pair<int,int>> q;
	q.push(make_pair(x,y));
	int npoint=500;
    while(!q.empty()&&reg_size < depth.total()&&npoint)
    {
		x = q.front().first;
		y = q.front().second;
		q.pop();
		npoint--;
        for (int j(0); j< 4; j++)
        {
            //Calculate the neighbour coordinate
            int xn = x + neigb.at<float>(j,0);
            int yn = y + neigb.at<float>(j,1);

            bool ins = ((xn >= 0) && (yn >= 0) && (xn < im.cols) && (yn < im.rows));
            if (ins && (J.at<float>(yn,xn) == 0.)&&(im.at<uchar>(yn,xn)!=objectId))
            {
				int xIni = max(0,xn-1);
                int yIni = max(0,yn-1);
                int xEnd = min(depth.cols,xn+1);
                int yEnd = min(depth.rows,yn+1);
				cv::Mat patch_depth = depth.rowRange(yIni,yEnd).colRange(xIni,xEnd);
				cv::Mat patch_gray = gray.rowRange(yIni,yEnd).colRange(xIni,xEnd);
                cv::Mat mean, stddev;
                cv::meanStdDev(patch_depth,mean,stddev);
                double _stddev = stddev.at<double>(0,0);
                double var_depth = _stddev*_stddev;
                cv::meanStdDev(patch_gray,mean,stddev);
				_stddev = stddev.at<double>(0,0);
                double var_gray = _stddev*_stddev;
                if (var_depth < dThresh&&var_gray<gThresh){
					q.push(make_pair(xn,yn));
					im.at<uchar>(yn,xn) = objectId;
					J.at<float>(yn,xn) = 1.;
				}
            }
        }
    }
}
const Mat Superpixels::sobel = (Mat_<float>(3,3) << -1/16., -2/16., -1/16., 0, 0, 0, 1/16., 2/16., 1/16.);

Superpixels::Superpixels(Mat& img, float m, float S){
    this->img = img.clone();
    this->m = m;
    if(S == USE_DEFAULT_S){
        this->nx = 15; // cols
        this->ny = 15; // rows
        this->dx = img.cols / float(nx); //steps
        this->dy = img.rows / float(ny);
        this->S = (dx + dy + 1)/2; // default window size
    }
    else
        this->S = S;
        
    calculateSuperpixels();
}

Mat Superpixels::viewSuperpixels(){    

	// Draw boundaries on original image
	vector<Mat> rgb(3);
	split(this->img_f, rgb);
	for (int i = 0; i < 3; i++){
        rgb[i] = rgb[i].mul(this->show);
    }
    
    Mat output = this->img_f.clone();
	merge(rgb, output);

    output = 255 * output;
    output.convertTo(output, CV_8UC3);
    
    return output;
}

Mat Superpixels::colorSuperpixels(){
    
    int n = nx * ny;
    vector<Vec3b> avg_colors(n);
    vector<int> num_pixels(n);
    
    vector<long> b(n), g(n), r(n);
    
    for(int y = 0; y < (int) labels.rows; ++y){
        for(int x = 0; x < (int) labels.cols; ++x){

            Vec3b pix = img.at<Vec3b>(y, x);
            int lbl = labels.at<int>(y, x);
            
            b[lbl] += (int) pix[0];
            g[lbl] += (int) pix[1];
            r[lbl] += (int) pix[2];
            
            ++num_pixels[lbl];
        }
    }

    for(int i = 0; i < n; ++i){
        int num = num_pixels[i];
        avg_colors[i] = Vec3b(b[i] / num, g[i] / num, r[i] / num);
    }
    
    Mat output = this->img.clone();
    for(int y = 0; y < (int) output.rows; ++y){
        for(int x = 0; x < (int) output.cols; ++x){
            int lbl = labels.at<int>(y, x);
            if(num_pixels[lbl])
                output.at<Vec3b>(y, x) = avg_colors[lbl];
        }
    }
    
    return output;
}

vector<Point> Superpixels::getCenters(){
    return centers;
}

Mat Superpixels::getLabels(){
    return labels;
}

// map<int, Point> Superpixels::getCentersMap(){

//     map<int, Point> out;
//     for(int i = 0; i < (int) centers.size(); ++i){
//         Point p = centers[i];
//         int lbl = labels.at<int>(p);
//         out[lbl] = p;
//     }
//     return out;
// }

void Superpixels::calculateSuperpixels(){

    // Scale img to [0,1]
    this->img.convertTo(this->img_f, CV_32F, 1/255.);
    // Convert to l-a-b colorspace
	cvtColor(this->img_f, this->img_lab, COLOR_BGR2Lab);

    int n = nx * ny;
    int w = img.cols;
    int h = img.rows;
    
	for (int i = 0; i < ny; i++) {
		for (int j = 0; j < nx; j++) {
			this->centers.push_back( Point2f(j*dx+dx/2, i*dy+dy/2));
		}
	}

	// Initialize labels and distance maps
	vector<int> label_vec(n);
	for (int i = 0; i < n; i++)
        label_vec[i] = i*255*255/n;

	Mat labels = -1*Mat::ones(this->img_lab.size(), CV_32S);
	Mat dists = -1*Mat::ones(this->img_lab.size(), CV_32F);
	Mat window;
	Point2i p1, p2;
	Vec3f p1_lab, p2_lab;

	// Iterate 10 times. In practice more than enough to converge
	for (int i = 0; i < 10; i++) {
		// For each center...
		for (int c = 0; c < n; c++)
            {
                int label = label_vec[c];
                p1 = centers[c];
                int xmin = max<int>(p1.x - S, 0);
                int ymin = max<int>(p1.y - S, 0);
                int xmax = min<int>(p1.x + S, w - 1);
                int ymax = min<int>(p1.y + S, h - 1);

                // Search in a window around the center
                window = this->img_f(Range(ymin, ymax), Range(xmin, xmax));
			
                // Reassign pixels to nearest center
                for (int i = 0; i < window.rows; i++) {
                    for (int j = 0; j < window.cols; j++) {
                        p2 = Point2i(xmin + j, ymin + i);
                        float d = dist(p1, p2);
                        float last_d = dists.at<float>(p2);
                        if (d < last_d || last_d == -1) {
                            dists.at<float>(p2) = d;
                            labels.at<int>(p2) = label;
                        }
                    }
                }
            }
	}

    // Store the labels for each pixel
    this->labels = labels.clone();
    this->labels =  n * this->labels / (255 * 255);

    // Calculate superpixel boundaries
	labels.convertTo(labels, CV_32F);

	Mat gx, gy, grad;
	filter2D(labels, gx, -1, sobel);
	filter2D(labels, gy, -1, sobel.t());
	magnitude(gx, gy, grad);
	grad = (grad > 1e-4)/255;
    Mat show = 1 - grad;
    show.convertTo(show, CV_32F);
        
    // Store the result
    this->show = show.clone();
}

float Superpixels::dist(Point p1, Point p2){
    Vec3f p1_lab = this->img_lab.at<Vec3f>(p1);
    Vec3f p2_lab = this->img_lab.at<Vec3f>(p2);
    
    float dl = p1_lab[0] - p2_lab[0];
	float da = p1_lab[1] - p2_lab[1];
	float db = p1_lab[2] - p2_lab[2];

	float d_lab = sqrtf(dl*dl + da*da + db*db);

	float dx = p1.x - p2.x;
	float dy = p1.y - p2.y;

	float d_xy = sqrtf(dx*dx + dy*dy);

	return d_lab + m/S * d_xy;
}


cv::Mat lime::lime_enhance(cv::Mat &src)
{
	cv::Mat img_norm;
	channel = src.channels();
	src.convertTo(img_norm, CV_32F, 1 / 255.0, 0);

	cv::Size sz(img_norm.size());
	cv::Mat out(sz, CV_32F, cv::Scalar::all(0.0));

	auto gammT = out.clone();

	if (channel == 3)
	{

		Illumination(img_norm, out);
		Illumination_filter(out, gammT);

		//lime
		std::vector<cv::Mat> img_norm_rgb;
		cv::Mat img_norm_b, img_norm_g, img_norm_r;

		cv::split(img_norm, img_norm_rgb);

		img_norm_g = img_norm_rgb.at(0);
		img_norm_b = img_norm_rgb.at(1);
		img_norm_r = img_norm_rgb.at(2);

		cv::Mat one = cv::Mat::ones(sz, CV_32F);

		float nameta = 0.7;
		auto g = 1 - ((one - img_norm_g) - (nameta * (one - gammT))) / gammT;
		auto b = 1 - ((one - img_norm_b) - (nameta * (one - gammT))) / gammT;
		auto r = 1 - ((one - img_norm_r) - (nameta * (one - gammT))) / gammT;

		cv::Mat g1, b1, r1;

		//TODO <=1
		threshold(g, g1, 0.0, 0.0, 3);
		threshold(b, b1, 0.0, 0.0, 3);
		threshold(r, r1, 0.0, 0.0, 3);

		img_norm_rgb.clear();
		img_norm_rgb.push_back(g1);
		img_norm_rgb.push_back(b1);
		img_norm_rgb.push_back(r1);

		cv::merge(img_norm_rgb,out_lime);
		out_lime.convertTo(out_lime,CV_8U,255);

	}
	else if(channel == 1)
	{
		Illumination_filter(img_norm, gammT);
		cv::Mat one = cv::Mat::ones(sz, CV_32F);
		float nameta = 0.7;
		//std::cout<<img_norm.at<float>(1,1)<<std::endl;
		auto out = 1 - ((one - img_norm) - (nameta * (one - gammT))) / gammT;

		threshold(out, out_lime, 0.0, 0.0, 3);

		out_lime.convertTo(out_lime,CV_8UC1,255);

	}

	else
	{
		std::cout<<"There is a problem with the channels"<<std::endl;
		exit(-1);
	}
	return out_lime.clone();
}

void lime::Illumination_filter(cv::Mat& img_in,cv::Mat& img_out)
{
	int ksize = 5;
	//mean filter
	blur(img_in,img_out,cv::Size(ksize,ksize));
	//GaussianBlur(img_in,img_mean_filter,Size(ksize,ksize),0,0);

	//gamma
	int row = img_out.rows;
	int col = img_out.cols;
	float tem;
	float gamma = 0.8;
	for(int i=0;i<row;i++)
	{

		for(int j=0;j<col;j++)
		{
			tem = pow(img_out.at<float>(i,j),gamma);
			tem = tem <= 0 ? 0.0001 : tem;  //  double epsolon = 0.0001;
			tem = tem > 1 ? 1 : tem;

			img_out.at<float>(i,j) = tem;

		}
	}

}
void lime::Illumination(cv::Mat& src,cv::Mat& out)
{
	int row = src.rows, col = src.cols;

	for(int i=0;i<row;i++)
	{
		for(int j=0;j<col;j++)
		{
			out.at<float>(i,j) = lime::compare(src.at<cv::Vec3f>(i,j)[0],
												src.at<cv::Vec3f>(i,j)[1],
												src.at<cv::Vec3f>(i,j)[2]);
		}

	}

}