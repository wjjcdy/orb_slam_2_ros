/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId=0;

// 关键帧构造函数，有普通帧构建
/*
input:当前帧
地图

*/
KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap)
{
    // 关键帧ID
    mnId=nNextId++;

    // 分配该帧的栅格化后的空间
    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    SetPose(F.mTcw);    
}

// 将特征点计算成bow，如果已计算过，则无需计算
void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

// 设置关键帧的位姿
/*
已知：Xc为相机坐标系，Xw为世界坐标系Xc = R * Xw + t
则：Xw = R^-1  Xc - R^-1  * t,
因为光心坐标相对于相机坐标为原点，所以Xc = 0,
所以得到光心在世界坐标系中的坐标Ow为：Ow = -R^-1 * t 
又因为：旋转矩阵R为正交矩阵，所以R^T = R^-1则：Ow = -R^T * t
*/
void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    // 旋转矩阵
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    // 平移矩阵
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    // 旋转矩阵的转置矩阵
    cv::Mat Rwc = Rcw.t();
    // 光心在世界坐标系下的坐标
    Ow = -Rwc*tcw;

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    // 可以认为立体空间相机的中心是左相机右移半个基线距离
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    Cw = Twc*center;
}

// 获取当前帧位姿
cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

// 获取当前帧位置的逆
cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

// 获取摄像机光心位置
cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

// 获取双目摄像机立体光心位置
cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}

// 获取旋转矩阵
cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

// 获取平移矩阵
cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

// 增加一个共视关系，即该关键帧与另一个关键帧的链接
/*input:
另一个关键帧
weight 权重， 应该是两个关键帧共同看到特征点的个数
*/
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    //更新链接中keyframe的顺序
    UpdateBestCovisibles();
}

// 每添加一个新的连接，则更新共视关系的顺序，即根据共视点的个数进行排序
void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    //遍历该帧与所有存在连接的关键帧， 并构成对，且将权重即个数放在前面
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));
    // 根据权重排列
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    // 按顺序将权重和frame放入list中
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    // 经排过顺序的连接
    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}

// 获取所有和该关键帧连接的关键帧set
set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

// 获取经排序后的共视的关键帧的队列
vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

// 仅获取前N个共视的关键帧队列（即存在更多共同特征点的前N个）
vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

// 获取公视keyframe中存在相同特征点个数大于W的所有关键帧队列
vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    // 查找大于w的迭代位置
    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        // 取出个数大于w的所有队列
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

// 获得和指定keyframe之间共同的特征点个数
int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

// 添加特征点
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

// 删除指定索引的特征点
void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

// 删除指定的特征点
void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    // 先查找此特征点在该frame下的索引
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

// 替换指定位置的特征点
void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

// 读出此帧中所有的特征点
set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

// 跟踪特征点被关键帧看到的次数大于等于minobs的mappoint个数
int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    // 设置的最少观测个数是否为0
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                // 如设置值不为0，当特征点
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

// 获取该关键帧中所有特征点
vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

// 根据索引号返回特征点
MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

//更新关键帧间的链接
void KeyFrame::UpdateConnections()
{
    // 该帧与其他关键帧共视点个数
    map<KeyFrame*,int> KFcounter;

    // 获取该帧中所有特征点
    vector<MapPoint*> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    // 针对每一个特征点共视的其他关键帧，统计其他帧与当前帧共视特征点个数。
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        // 遍历每个特征点
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        // 此特征点所有的观测帧
        map<KeyFrame*,size_t> observations = pMP->GetObservations();

        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            // 此特征点被当前帧观测到，不用计数
            if(mit->first->mnId==mnId)
                continue;
            // 记录另一个关键帧观测到的特征点个数
            KFcounter[mit->first]++;
        }
    }

    // 以上简单的方法就是，遍历当前帧中每一个特征点，然后根据每个特征点中的观测关系，提取可以
    // 看到此特征点的关键帧，然后进行统计个数

    // This should not happen
    // 没有一个关键帧和此关键帧共视，基本不可能
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    // 共视权重（共视的特征点个数）
    int th = 15;

    // 创建对容器，并开辟空间
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    // 遍历和当前帧存在共视点其他keyframe
    // 
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        // 查找与当前帧最大的权重的keyframe，即存在最多共视特征点的keyframe
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        // 当共视点个数大于th时，则保存
        if(mit->second>=th)
        {
            // 记录满足条件的链接
            vPairs.push_back(make_pair(mit->second,mit->first));
            // 将两个关键帧链接起来
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    // 如果共视的特征点数全都不满足最小个数，则仅保留最大的权重的对即可
    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    // 按照权重排序
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    // 重新将权重和keyframe记录为list
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;
        // 排过序的链接关键帧
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        // 不是第一帧，且第一次放入树中
        if(mbFirstConnection && mnId!=0)
        {
            // 获取排序后的第一个关键帧作为parent
            mpParent = mvpOrderedConnectedKeyFrames.front();
            // 将本关键帧添加为子叶
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

// 关键帧添加叶子
void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

// 删除叶子
void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

// 修改本帧的父节点
void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

// 获取所有的叶子
set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

// 获取父节点
KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

// 判断一个关键帧是否在子叶节点中
bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

// 添加闭环的边，即和本帧有共视的关键帧
void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

// 获取闭环的边
set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

// 闭环节点不能被擦除标志，表明在将在闭环处理中
void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

// 设置闭环节点队列的擦除标志
void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        // 如果闭环边为空，直接设置false即可
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    // 如果需要删除直接删除
    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

// 删除此关键帧
// 1.遍历将每一个keyframe与此keyframe有connect的从容器中删除
// 2. 遍历每一个特征点，删除该特征点与本keyframe的观测
// 3. 清除此keyframe与其他keyframe的链接
// 4. 为此帧的每一个子节点重新安排新的父节点
// 5. 从地图中删除此keyframe
// 6. 从字典中删除此keyframe
void KeyFrame::SetBadFlag()
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        // 若为第一个关键帧，则不删除
        if(mnId==0)
            return;
        //如果闭环的边未被删除，应该先设置闭环将被擦除标志
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    // 遍历将每一个keyframe与此keyframe有connect的从容器中删除
    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);

    // 遍历每一个特征点，删除该特征点与本keyframe的观测
    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);

    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        // 清除此keyframe与其他keyframe的链接
        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        // 当前帧的父节点即为当前帧叶子的候选点
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        // 由于此关键帧将被删除，则此关键帧的所有叶子均会没有父节点。
        // 因此需要为每一个子叶点重新给定新的父节点；
        // 每个子叶的候选父节点包括，本帧的父节点和其他子叶已找到的父节点
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            // 遍历每一个叶节点
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                // 读取叶节点
                KeyFrame* pKF = *sit;
                // 已被删除，无需考虑
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                // 获取该关键帧中链接的所有其他keyframe（已根据权重排序过）
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                // 遍历一叶节点的每个连接即边
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    // 遍历所有候选父节点
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        // 如果候选的父节点在此叶节点的已排序的链接边中，且找到最大权重的候选父节点
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            // 该帧的一个叶节点找到了父节点
            if(bContinue)
            {
                // 更换父节点
                pC->ChangeParent(pP);
                // 找到的一个父节点可成为候选节点
                sParentCandidates.insert(pC);
                // 擦除此帧的已找到新的父节点的叶节点
                mspChildrens.erase(pC);
            }
            // 叶节点未找到新的父节点
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        // 如果不能在候选父节点中找到父节点，则直接将此帧的父节点直接当做此叶节点的父节点
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }

        // 最后从父节点删除此帧
        mpParent->EraseChild(this);
        // 相对于父节点的相对pose
        mTcp = Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }

    // 将地图中删除此keyframe
    mpMap->EraseKeyFrame(this);
    // 将关键帧生成的字典也删除此帧
    mpKeyFrameDB->erase(this);
}

// 判断是否为删除的keyframe
bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

// 从该帧中删除一个连接
void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        // 从该帧的连接中查找是否存在，如果存在，则删除
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    //更新链接中keyframe的顺序
    if(bUpdate)
        UpdateBestCovisibles();
}

// 获取设置区域内的特征点
vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

// 判断某个坐标是否在本keyframe中
bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

//去畸变
cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}

// 求取keyframe中地图点q 位置深度
float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        //获取所有特征点
        vpMapPoints = mvpMapPoints;
        // 获取该keyframe的pose
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    //开辟特征点深度信息空间
    vDepths.reserve(N);
    //获取旋转矩阵
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    // 获取平移矩阵
    float zcw = Tcw_.at<float>(2,3);
    // 遍历每个点
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            // 获取世界坐标
            cv::Mat x3Dw = pMP->GetWorldPos();
            // 获取相对于摄像机的深度
            float z = Rcw2.dot(x3Dw)+zcw;
            // 放入vector
            vDepths.push_back(z);
        }
    }

    // 排序
    sort(vDepths.begin(),vDepths.end());

    // 
    return vDepths[(vDepths.size()-1)/q];
}

// map serialization addition
// Default serializing Constructor
KeyFrame::KeyFrame():
    mnFrameId(0),  mTimeStamp(0.0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(0.0), mfGridElementHeightInv(0.0),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(0.0), fy(0.0), cx(0.0), cy(0.0), invfx(0.0), invfy(0.0),
    mbf(0.0), mb(0.0), mThDepth(0.0), N(0), mnScaleLevels(0), mfScaleFactor(0),
    mfLogScaleFactor(0.0),
    mnMinX(0), mnMinY(0), mnMaxX(0),
    mnMaxY(0)
{}
template<class Archive>
void KeyFrame::serialize(Archive &ar, const unsigned int version)
{
    // no mutex needed vars
    ar & nNextId;
    ar & mnId;
    ar & const_cast<long unsigned int &>(mnFrameId);
    ar & const_cast<double &>(mTimeStamp);
    // Grid related vars
    ar & const_cast<int &>(mnGridCols);
    ar & const_cast<int &>(mnGridRows);
    ar & const_cast<float &>(mfGridElementWidthInv);
    ar & const_cast<float &>(mfGridElementHeightInv);
    // Tracking related vars
    ar & mnTrackReferenceForFrame & mnFuseTargetForKF;
    // LocalMaping related vars
    ar & mnBALocalForKF & mnBAFixedForKF;
    // KeyFrameDB related vars
    ar & mnLoopQuery & mnLoopWords & mLoopScore & mnRelocQuery & mnRelocWords & mRelocScore;
    // LoopClosing related vars
    ar & mTcwGBA & mTcwBefGBA & mnBAGlobalForKF;
    // calibration parameters
    ar & const_cast<float &>(fx) & const_cast<float &>(fy) & const_cast<float &>(cx) & const_cast<float &>(cy);
    ar & const_cast<float &>(invfx) & const_cast<float &>(invfy) & const_cast<float &>(mbf);
    ar & const_cast<float &>(mb) & const_cast<float &>(mThDepth);
    // Number of KeyPoints;
    ar & const_cast<int &>(N);
    // KeyPoints, stereo coordinate and descriptors
    ar & const_cast<std::vector<cv::KeyPoint> &>(mvKeys);
    ar & const_cast<std::vector<cv::KeyPoint> &>(mvKeysUn);
    ar & const_cast<std::vector<float> &>(mvuRight);
    ar & const_cast<std::vector<float> &>(mvDepth);
    ar & const_cast<cv::Mat &>(mDescriptors);
    // Bow
    ar & mBowVec & mFeatVec;
    // Pose relative to parent
    ar & mTcp;
    // Scale related
    ar & const_cast<int &>(mnScaleLevels) & const_cast<float &>(mfScaleFactor) & const_cast<float &>(mfLogScaleFactor);
    ar & const_cast<std::vector<float> &>(mvScaleFactors) & const_cast<std::vector<float> &>(mvLevelSigma2) & const_cast<std::vector<float> &>(mvInvLevelSigma2);
    // Image bounds and calibration
    ar & const_cast<int &>(mnMinX) & const_cast<int &>(mnMinY) & const_cast<int &>(mnMaxX) & const_cast<int &>(mnMaxY);
    ar & const_cast<cv::Mat &>(mK);

    // mutex needed vars, but don't lock mutex in the save/load procedure
    {
        unique_lock<mutex> lock_pose(mMutexPose);
        ar & Tcw & Twc & Ow & Cw;
    }
    {
        unique_lock<mutex> lock_feature(mMutexFeatures);
        ar & mvpMapPoints; // hope boost deal with the pointer graph well
    }
    // BoW
    ar & mpKeyFrameDB;
    // mpORBvocabulary restore elsewhere(see SetORBvocab)
    {
        // Grid related
        unique_lock<mutex> lock_connection(mMutexConnections);
        ar & mGrid & mConnectedKeyFrameWeights & mvpOrderedConnectedKeyFrames & mvOrderedWeights;
        // Spanning Tree and Loop Edges
        ar & mbFirstConnection & mpParent & mspChildrens & mspLoopEdges;
        // Bad flags
        ar & mbNotErase & mbToBeErased & mbBad & mHalfBaseline;
    }
    // Map Points
    ar & mpMap;
    // don't save mutex
}
template void KeyFrame::serialize(boost::archive::binary_iarchive&, const unsigned int);
template void KeyFrame::serialize(boost::archive::binary_oarchive&, const unsigned int);

} //namespace ORB_SLAM
