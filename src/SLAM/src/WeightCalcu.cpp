#include "WeightCalcu.h"

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
namespace ORB_SLAM3
{
    const float TDistributionScaleEstimator::INITIAL_SIGMA = 5.0f;
    const float TDistributionScaleEstimator::DEFAULT_DOF = 5.0f;

    TDistributionScaleEstimator::TDistributionScaleEstimator(const float dof) : initial_sigma(INITIAL_SIGMA)
    {
        configure(dof);
    }

    void TDistributionScaleEstimator::configure(const float &param)
    {
        dof = param;
    }

    float TDistributionScaleEstimator::compute(const cv::Mat &errors) const
    {
        float initial_lamda = 1.0f / (initial_sigma * initial_sigma);

        float num = 0.0f;
        float lambda = initial_lamda;

        int iterations = 0;

        do
        {
            iterations += 1;
            initial_lamda = lambda;
            num = 0.0f;
            lambda = 0.0f;

            const float *data_ptr = errors.ptr<float>();

            for (size_t idx = 0; idx < errors.size().area(); ++idx, ++data_ptr)
            {
                const float &data = *data_ptr;

                if (std::isfinite(data))
                {
                    num += 1.0f;
                    lambda += data * data * ((dof + 1.0f) / (dof + initial_lamda * data * data));
                }
            }

            lambda /= num;
            lambda = 1.0f / lambda;
        } while (std::abs(lambda - initial_lamda) > 1e-3);

        return std::sqrt(1.0f / lambda);
    }

    const float TDistributionInfluenceFunction::DEFAULT_DOF = 5.0f;

    TDistributionInfluenceFunction::TDistributionInfluenceFunction(const float dof)
    {
        configure(dof);
    }

    inline float TDistributionInfluenceFunction::value(const float &x) const
    {

        return ((dof + 1.0f) / (dof + (x * x)));
    }

    void TDistributionInfluenceFunction::configure(const float &param)
    {
        dof = param;
        normalizer = dof / (dof + 1.0f);
    }

    WeightCalculation::WeightCalculation() : scale_(1.0f)
    {
    }

    void WeightCalculation::calculateScale(const cv::Mat &errors)
    {
        // some scale estimators might return 0
        scale_ = std::max(scaleEstimator_->compute(errors), 0.001f);
    }

    float WeightCalculation::calculateWeight(const float error) const
    {
        return influenceFunction_->value(error / scale_);

    }

    void WeightCalculation::calculateWeights(const cv::Mat &errors, cv::Mat &weights)
    {
        weights.create(errors.size(), errors.type());

        cv::Mat scaled_errors = errors / scale_;
        const float *err_ptr = scaled_errors.ptr<float>();
        float *weight_ptr = weights.ptr<float>();

        for (size_t idx = 0; idx < errors.size().area(); ++idx, ++err_ptr, ++weight_ptr)
        {
            if (std::isfinite(*err_ptr))
            {
                *weight_ptr = influenceFunction_->value(*err_ptr);
            }
            else
            {
                *weight_ptr = 0.0f;
            }
        }
    }
};