#include "Map.h"
#include "Cluster.h"
using namespace std;
namespace ORB_SLAM3
{
    void Map::AddCluster(Cluster* pCluster)
    {
        unique_lock<mutex> lock(mMutexMap);
        mspClusters.insert(pCluster);
    }
    void Map::EraseCluster(Cluster* pCluster)
    {
        unique_lock<mutex> lock(mMutexMap);
        mspClusters.erase(pCluster);
    }
    Cluster* Map::GetCluster(int cluster_id)
    {
        unique_lock<mutex> lock(mMutexMap);
        for (auto pCluster : mspClusters)
        {
            if(pCluster->mnId != cluster_id)
                continue;
            return pCluster;
        }
        return NULL;
    }
    std::vector<Cluster*> Map::GetAllClusters()
    {
        unique_lock<mutex> lock(mMutexMap);
        return vector<Cluster*>(mspClusters.begin(), mspClusters.end());
    }
}