/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"

#include <opencv2/core/core.hpp>
#include <mutex>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>

namespace ORB_SLAM3
{

    class KeyFrame;
    class Map;
    class Frame;
    class Cluster;

    class MapPoint
    {
        // [serialize setting]
        template <class Archive>
        void serializeMatric(Archive &ar, cv::Mat &mat, const unsigned int version)
        {
            int cols, rows, type;
            bool continuous;
            if (Archive::is_saving::value)
            {
                cols = mat.cols;
                rows = mat.rows;
                type = mat.type();
                continuous = mat.isContinuous();
            }
            ar &cols &rows &type &continuous;
            if (Archive::is_loading::value)
            {
                mat.create(rows, cols, type);
            }
            if (continuous)
            {
                const unsigned int data_size = rows * cols * mat.elemSize();
                ar &boost::serialization::make_array(mat.ptr(), data_size);
            }
            else
            {
                const unsigned int row_size = cols * mat.elemSize();
                for (int i = 0; i < rows; i++)
                {
                    ar &boost::serialization::make_array(mat.ptr(i),row_size);
                }
            }
        }

        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            ar &mnId;
            ar &mnFirstKFid;
            ar &mnFirstFrame;
            ar &nObs;
            // Variables used by the tracking
            ar &mTrackProjX;
            ar &mTrackProjY;
            ar &mTrackDepth;
            ar &mTrackDepthR;
            ar &mTrackProjXR;
            ar &mTrackProjYR;
            ar &mbTrackInView;
            ar &mbTrackInViewR;
            ar &mnTrackScaleLevel;
            ar &mnTrackScaleLevelR;
            ar &mTrackViewCos;
            ar &mTrackViewCosR;
            ar &mnTrackReferenceForFrame;
            ar &mnLastFrameSeen;

            // Variables used by local mapping
            ar &mnBALocalForKF;
            ar &mnFuseCandidateForKF;

            // Variables used by loop closing and merging
            ar &mnLoopPointForKF;
            ar &mnCorrectedByKF;
            ar &mnCorrectedReference;
            serializeMatrix(ar, mPosGBA, version);
            ar &mnBAGlobalForKF;
            ar &mnBALocalForMerge;
            serializeMatrix(ar, mPosMerge, version);
            serializeMatrix(ar, mNormalVectorMerge, version);

            // Protected variables
            serializeMatrix(ar, mWorldPos, version);
            // ar & BOOST_SERIALIZATION_NVP(mBackupObservationsId);
            ar &mBackupObservationsId1;
            ar &mBackupObservationsId2;
            serializeMatrix(ar, mNormalVector, version);
            serializeMatrix(ar, mDescriptor, version);
            ar &mBackupRefKFId;
            ar &mnVisible;
            ar &mnFound;

            ar &mbBad;
            ar &mBackupReplacedId;

            ar &mfMinDistance;
            ar &mfMaxDistance;
        }
    public:
        MapPoint();

        MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap);
        MapPoint(const double invDepth, cv::Point2f uv_init, KeyFrame *pRefKF, KeyFrame *pHostKF, Map *pMap);
        MapPoint(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF);

        void SetWorldPos(const cv::Mat &Pos);

        cv::Mat GetWorldPos();

        cv::Mat GetNormal();

        cv::Matx31f GetWorldPos2();

        cv::Matx31f GetNormal2();

        KeyFrame *GetReferenceKeyFrame();

        std::map<KeyFrame *, std::tuple<int, int>> GetObservations();
        int Observations();

        void AddObservation(KeyFrame *pKF, int idx);
        void EraseObservation(KeyFrame *pKF);

        std::tuple<int, int> GetIndexInKeyFrame(KeyFrame *pKF);
        bool IsInKeyFrame(KeyFrame *pKF);

        void SetBadFlag();
        bool isBad();

        void Replace(MapPoint *pMP);
        MapPoint *GetReplaced();

        void IncreaseVisible(int n = 1);
        void IncreaseFound(int n = 1);
        float GetFoundRatio();
        inline int GetFound()
        {
            return mnFound;
        }

        void ComputeDistinctiveDescriptors();

        cv::Mat GetDescriptor();

        void UpdateNormalAndDepth();
        void SetNormalVector(cv::Mat &normal);

        float GetMinDistanceInvariance();
        float GetMaxDistanceInvariance();
        int PredictScale(const float &currentDist, KeyFrame *pKF);
        int PredictScale(const float &currentDist, Frame *pF);

        Map *GetMap();
        void UpdateMap(Map *pMap);

        // [semantic]
        bool IsDynamicMapPoint();
        void SetMovingProbability(const float &in_mp);
        float GetMovingProbability();
        void PrintObservations();
        void PreSave(set<KeyFrame *> &spKF, set<MapPoint *> &spMP);
        void PostLoad(map<long unsigned int, KeyFrame *> &mpKFid, map<long unsigned int, MapPoint *> &mpMPid);

    public:
        long unsigned int mnId;
        static long unsigned int nNextId;
        long int mnFirstKFid;
        long int mnFirstFrame;
        int nObs;

        // Variables used by the tracking
        float mTrackProjX;
        float mTrackProjY;
        float mTrackDepth;
        float mTrackDepthR;
        float mTrackProjXR;
        float mTrackProjYR;
        bool mbTrackInView, mbTrackInViewR;
        int mnTrackScaleLevel, mnTrackScaleLevelR;
        float mTrackViewCos, mTrackViewCosR;
        long unsigned int mnTrackReferenceForFrame;
        long unsigned int mnLastFrameSeen;

        // Variables used by local mapping
        long unsigned int mnBALocalForKF;
        long unsigned int mnFuseCandidateForKF;

        // Variables used by loop closing
        long unsigned int mnLoopPointForKF;
        long unsigned int mnCorrectedByKF;
        long unsigned int mnCorrectedReference;
        cv::Mat mPosGBA;
        long unsigned int mnBAGlobalForKF;
        long unsigned int mnBALocalForMerge;

        // Variable used by merging
        cv::Mat mPosMerge;
        cv::Mat mNormalVectorMerge;

        // Fopr inverse depth optimization
        double mInvDepth;
        double mInitU;
        double mInitV;
        KeyFrame *mpHostKF;

        static std::mutex mGlobalMutex;

        unsigned int mnOriginMapId;

        // [semantic]
        float mMovingProbability;
        float mStaticProbability;
        int mnObservedStatic;
        int mnObservedDynamic;
        long unsigned int mnBASemanticForKF;
        long unsigned int mnSemanticTrackReferenceForFrame;
        int mnObjectId=-1;
        bool mbGeometryDynamic=false;
        Cluster* mpCluster;
        // Reference KeyFrame
        KeyFrame *mpRefKF;
    protected:
        // Position in absolute coordinates
        cv::Mat mWorldPos;
        cv::Matx31f mWorldPosx;

        // Keyframes observing the point and associated index in keyframe
        std::map<KeyFrame *, std::tuple<int, int>> mObservations;

        // Mean viewing direction
        cv::Mat mNormalVector;
        cv::Matx31f mNormalVectorx;

        // Best descriptor to fast matching
        cv::Mat mDescriptor;


        // Tracking counters
        int mnVisible;
        int mnFound;

        // Bad flag (we do not currently erase MapPoint from memory)
        bool mbBad;
        MapPoint *mpReplaced;

        // Scale invariance distances
        float mfMinDistance;
        float mfMaxDistance;

        Map *mpMap;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;
        std::mutex mMutexMap;

        // For save relation without pointer, this is necessary for save/load function
        std::map<long unsigned int, int> mBackupObservationsId1;
        std::map<long unsigned int, int> mBackupObservationsId2;
        long unsigned int mBackupRefKFId;
        long long int mBackupReplacedId;

    public:
        // Object SLAM by Jingwen
        bool in_any_object;  // Flag indicating the point belongs to an object
        int object_id; // ID of the associated object
        int keyframe_id_added_to_object; // KF id when added to object
        bool mbOutlier;  // Outlier for object
        void SetOutlierFlag();
        bool isOutlier();
    };

} // namespace ORB_SLAM

#endif // MAPPOINT_H
