#ifndef _WEIGHT_CALCU_H_
#define _WEIGHT_CALCU_H_

#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{
    class ScaleEstimator
    {
    public:
        virtual ~ScaleEstimator(){};
        virtual float compute(const cv::Mat &errors) const = 0;
        virtual void configure(const float &param){};
    };

    class TDistributionScaleEstimator : public ScaleEstimator
    {
    public:
        TDistributionScaleEstimator(const float dof = DEFAULT_DOF);
        virtual ~TDistributionScaleEstimator(){};
        virtual float compute(const cv::Mat &errors) const;
        virtual void configure(const float &param);

        static const float DEFAULT_DOF;
        static const float INITIAL_SIGMA;

    protected:
        float dof;
        float initial_sigma;
    };

    class InfluenceFunction
    {
    public:
        virtual ~InfluenceFunction(){};
        virtual float value(const float &x) const = 0;
        virtual void configure(const float &param){};
    };
    class TDistributionInfluenceFunction : public InfluenceFunction
    {
    public:
        TDistributionInfluenceFunction(const float dof = DEFAULT_DOF);
        virtual ~TDistributionInfluenceFunction(){};
        virtual inline float value(const float &x) const;
        virtual void configure(const float &param);

        static const float DEFAULT_DOF;

    private:
        float dof;
        float normalizer;
    };
    
    class WeightCalculation
    {
    public:
        WeightCalculation();
        ScaleEstimator *scaleEstimator_ = new TDistributionScaleEstimator();
        InfluenceFunction *influenceFunction_ = new TDistributionInfluenceFunction();

        void calculateScale(const cv::Mat &errors);

        float calculateWeight(const float error) const;

        void calculateWeights(const cv::Mat &errors, cv::Mat &weights);

        float scale_;

    private:
    };
}

#endif // !_WEIGHT_CALCU_H
