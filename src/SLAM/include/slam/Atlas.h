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

#ifndef ATLAS_H
#define ATLAS_H

#include "Map.h"
#include "Frame.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "GeometricCamera.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"

#include <set>
#include <mutex>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>
#include "detect_3d_cuboid/detect_3d_cuboid.h"
class detect_3d_cuboid;
class line_lbd_detect;
namespace ORB_SLAM3
{
    class Viewer;
    class Map;
    class MapPoint;
    class KeyFrame;
    class KeyFrameDatabase;
    class Frame;
    class KannalaBrandt8;
    class Pinhole;
    class GeometricCamera;
    class Cluster;
    class Atlas
    {
        friend class boost::serialization::access;

        template <class Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            ar &mvpBackupCamKan;
            ar &mvpCameras;
            ar &Map::nNextId;
            ar &Frame::nNextId;
            ar &KeyFrame::nNextId;
            ar &MapPoint::nNextId;
            ar &GeometricCamera::nNextId;
            ar &mnLastInitKFidMap;
        }

    public:
        Atlas();
        Atlas(int initKFid); // When its initialization the first map is created
        ~Atlas();

        void CreateNewMap();
        void ChangeMap(Map *pMap);

        unsigned long int GetLastInitKFid();

        void SetViewer(Viewer *pViewer);

        // Method for change components in the current map
        void AddKeyFrame(KeyFrame *pKF);
        void AddMapPoint(MapPoint *pMP);
        void AddDynamicPoint(MapPoint *pMP,int objectid);
        void InitDynamicVec(int num,vector<vector<int>> &mvBbox);
        void SetSemanticLabel(int objectid,int semanticLabel);
        void SetDynamicPose(cv::Mat pose);
        cv::Mat GetDynamicPose();
        int GetDynamicNum();

        
        void AddCamera(GeometricCamera *pCam);

        /* All methods without Map pointer work on current map */
        void SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs);
        void InformNewBigChange();
        int GetLastBigChangeIdx();

        long unsigned int MapPointsInMap();
        long unsigned KeyFramesInMap();

        // Method for get data in current map
        std::vector<KeyFrame *> GetAllKeyFrames();
        std::vector<MapPoint *> GetAllMapPoints();
        std::vector<MapPoint *> GetReferenceMapPoints();

        vector<Map *> GetAllMaps();

        int CountMaps();

        void clearMap();

        void clearAtlas();

        Map *GetCurrentMap();

        void SetMapBad(Map *pMap);
        void RemoveBadMaps();

        bool isInertial();
        void SetInertialSensor();
        void SetImuInitialized();
        bool isImuInitialized();

        void SetKeyFrameDababase(KeyFrameDatabase *pKFDB);
        KeyFrameDatabase *GetKeyFrameDatabase();

        void SetORBVocabulary(ORBVocabulary *pORBVoc);
        ORBVocabulary *GetORBVocabulary();

        long unsigned int GetNumLivedKF();

        long unsigned int GetNumLivedMP();

        // Function for garantee the correction of serialization of this object
        void PreSave();
        void PostLoad();
        
        // void UpdateClusters(KeyFrame* pKF,Frame* pFrame,cv::Mat mask);
        // void UpdateClustersBbox(Frame* pFrame);
        // void UpdateMovingProb(KeyFrame* pKF,vector<cv::Mat> &masks);
        // void updateCurrentCluster(Frame* pFrame);
        // void UpdateMask(cv::Mat &imMask,cv::Mat &imFlow,cv::Mat &outMask);
        // void OptimizeClusters(KeyFrame *pKF);
        // vector<Cluster*> GetClusters();
        void UpdateClusters(KeyFrame *pKF, Frame *pFrame, cv::Mat mask,vector<ObjectSet> all_object_cuboids);

        void AssociateNewBBox(KeyFrame* pKF,cv::Mat mask,std::vector<std::vector<cuboid*>> all_object_cuboids);

        // std::vector<Cluster*> mvpCluster;
        KeyFrame* mpKFSemantic=nullptr;
        bool useKF=false;
        std::map<int,Cluster*> mClusterMapid;

        // static detect_3d_cuboid* detect_cuboid_obj;
        // static line_lbd_detect* line_lbd_ptr;
        // static Eigen::Matrix3d Kalib;
    protected:
        std::set<Map *> mspMaps;
        std::set<Map *> mspBadMaps;
        Map *mpCurrentMap;

        std::vector<GeometricCamera *> mvpCameras;
        std::vector<KannalaBrandt8 *> mvpBackupCamKan;
        std::vector<Pinhole *> mvpBackupCamPin;
        std::vector<Map *> mvpBackupMaps;
        
        std::mutex mMutexAtlas;

        unsigned long int mnLastInitKFidMap;

        Viewer *mpViewer;
        bool mHasViewer;

        // Class references for the map reconstruction from the save file
        KeyFrameDatabase *mpKeyFrameDB;
        ORBVocabulary *mpORBVocabulary;

    }; // class Atlas

} // namespace ORB_SLAM3

#endif // ATLAS_H
