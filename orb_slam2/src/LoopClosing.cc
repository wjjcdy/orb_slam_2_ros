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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(false)
{
    mnCovisibilityConsistencyTh = 3;
}

// 设置跟踪器
void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

// 设置局部地图构建器
void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        // 判断队列中是否存在新的KF
        if(CheckNewKeyFrames())
        {
            // Detect loop candidates and check covisibility consistency
            // 检测是否存在闭环，且共视连续
            if(DetectLoop())
            {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               // 查找到闭环，则计算闭环帧之间的pose转换矩阵
               if(ComputeSim3())
               {
                   // Perform loop fusion and pose graph optimization
                   // 全局闭环优化
                   CorrectLoop();
               }
            }
        }

        ResetIfRequested();

        if(CheckFinish())
            break;

        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

// 闭环检测
bool LoopClosing::DetectLoop()
{
    // 取出闭环queue中的最早的KF
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        // 取出的的这一帧，为防止被删除，则设置不能删除标志
        mpCurrentKF->SetNotErase();
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    // 如果当前帧距离上次闭环检测在10帧内，则可以删除，并退出
    // 但是每帧的都需要添加到BOW集合中
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
        //关键帧字典集合添加一帧
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    // 获取该帧的所有的共视的KF
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    // 该帧的词典向量
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    // 遍历所有共视帧
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        // 一一对比BOW相似度
        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        // 记录共视关系中最低评分
        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    // 提取可能闭环的候选帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    // 如果没有候选帧
    if(vpCandidateKFs.empty())
    {
        // 将此KF放入队列中
        mpKeyFrameDB->add(mpCurrentKF);
        // 记录连续帧的关联帧集合
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    // 判断是否存在闭环，则需要候选帧中具有一定连续性，即候选帧的应连续存在共视关系3次，则可作为闭环存在
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    // 遍历每个候选帧
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];

        // 获取一个候选帧的具有共视帧集合，构建一个group
        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        // 将当前候选帧也添加进去
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        // 将之前候选group进行遍历
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            // 提取之前的关联帧集合
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            // 遍历最新候选帧和之前的group是否存在相同，存在表明连续
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                // 如果之前的集合中和当前候选帧有重复，表明连续
                // 且仅需要一个存在相同的，即表明连续
                if(sPreviousGroup.count(*sit))
                {
                    // 闭环连续标志置位
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            // 如果连续性
            if(bConsistent)
            {
                // 和前面开始具有连续性的，提取其序号
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                // 新添加连续性group的序号
                int nCurrentConsistency = nPreviousConsistency + 1;
                // 如果未处理
                if(!vbConsistentGroup[iG])
                {
                    // 在连续性group中添加新的group
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    // 记录已处理过，避免在一组重复加入
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
                // 如果连续性group超过阈值3，且被处理过，则将该候选帧放入具有连续性候选帧的集合中
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    // 将候选帧放入队列中，即当连续性达到一定数量时，则候选帧放入一个集合中
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        // 不存在连续的关键帧的关联帧，即前面没有当前帧有关联，则重新记录
        if(!bConsistentForSomeGroup)
        {
            // 初始化候选帧的共视关联帧集合，即将序号改为0
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    // 更新关联帧集合
    mvConsistentGroups = vCurrentConsistentGroups;


    // Add Current Keyframe to database
    // 添加当前帧到database中
    mpKeyFrameDB->add(mpCurrentKF);

    // 如果连续候选帧为空,则当前帧可被删除
    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        // 表明存在闭环
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}

// 计算闭环下的转移矩阵（相似变换，即两帧间的相对位姿)
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3
    // 具有连续性的候选帧的个数
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    // 如果有足够的匹配，则设置一个相似变换求解器
    ORBmatcher matcher(0.75,true);

    // 针对每个候选变换帧开辟相似求解器
    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    // 地图点
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

    // 遍历每个连续的候选闭环帧
    for(int i=0; i<nInitialCandidates; i++)
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();

        // 此帧已被删除了，则标记
        if(pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        // 候选帧与当前帧进行BOW匹配
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);
        // 匹配的地图点少于20个，则认为匹配失败
        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            // 相似矩阵求解器
            // 如果mbFixScale为true，则是6DoFf优化（双目或rgbd），如果是false，则是7DoF优化（单目）
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
            // 采用随机采样法计算相似矩阵
            pSolver->SetRansacParameters(0.99,20,300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    // 标记是否匹配成功
    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    // 循环所有候选帧进行迭代
    while(nCandidates>0 && !bMatch)
    {
        // 遍历候选帧
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            // 每个候选帧迭代5次求解
            Sim3Solver* pSolver = vpSim3Solvers[i];
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            // 迭代5次，无结果，则剔除则此候选帧
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            // 如果有相似矩阵解
            if(!Scm.empty())
            {
                // 提取所有内点
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

                // 根据内点相似变换进行匹配
                cv::Mat R = pSolver->GetEstimatedRotation();
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);

                // 进行g2o优化
                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                // gScm 为优化后的pose
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                // If optimization is succesful stop ransacs and continue
                // 优化后pose后，如果内点个数超过20个，则表明匹配成功
                if(nInliers>=20)
                {
                    bMatch = true;
                    // 记录成功匹配的关键帧
                    mpMatchedKF = pKF;
                    // 世界坐标
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                    // 优化后的世界坐标
                    mg2oScw = gScm*gSmw;
                    mScw = Converter::toCvMat(mg2oScw);

                    // 更新loop 匹配的mappoint，并退出候选帧遍历
                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;
                }
            }
        }
    }

    // 匹配失败则退出
    if(!bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    // 检索在闭环帧相邻帧附被看见的mappoints
    // 搜索具有共视关联帧集合，包括本身自己
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    // 遍历每个关联帧，将所有关联帧中有效的mappoint进行集合
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    // 根据投影匹配搜索更多的当前帧匹配的mappoint
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    // 统计所有有效mappoint匹配个数
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }
    // 总匹配个数大于40时，则闭环求解成功
    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}

// 当闭环检测到时，则可进行闭环修正
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    // 开始修正闭环，需要暂停localmap线程
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    // 判断当前有没有在全局BA，如果有，需要暂停
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx = true;
        // 如果全局BA存在，则关闭线程
        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    // 等待local map功能确实停止
    while(!mpLocalMapper->isStopped())
    {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    // Ensure current keyframe is updated
    // 当前关键帧共视关系最新更新
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    // 搜索当前帧关联帧和本帧一起作为集合
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    //被优化的帧只有当前帧
    CorrectedSim3[mpCurrentKF]=mg2oScw;
    // 取出优化前的pose的逆
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        // 遍历当前帧以及具有共视关系的所有帧
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            cv::Mat Tiw = pKFi->GetPose();

            // 将所有共视帧根据优化后的闭环帧，修正其他共视帧
            if(pKFi!=mpCurrentKF)
            {
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                // 利用当前的优化后的pose更新其他关联帧的pose
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                // 存储优化后的pose
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            // 存储未优化的pose
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        // 根据优化后的帧的pose，修正mappoint
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            // 关键帧
            KeyFrame* pKFi = mit->first;
            // 修正后的pose
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();
            // 未修正的pose
            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            // 遍历一关键帧的所有mappoints
            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                // 根据keypose的pose转移，更新mappoint的pose
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));
                // 获取更新后的pose，并设置
                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                //更新mappoint 法向量和深度
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            // 更新每个keyframe的pose
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated（重复的）
        // 闭环后需要匹配的mappoint进行融合
        // 如果已经存在，则采用匹配的进行替代，如果不存在则新添加
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                // 如果对应位置的mappoint存在，则替换
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    // 如果当前帧对应位置不存在此mappoint，则添加对应关系
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    // 闭环中观测到mappoint融合到当前帧及其共视关系的关键帧中
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    // 经过mappoint fusion，则会产生新的connect，即loop带来的，将其提取放入loopconnection
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    // 遍历当前帧所有的connect keyframe
    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        // 获取每个connect keyframe所有排序后的共视帧集合（即第2层的排序后的共视帧）
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        // 更新二级关键帧关系
        pKFi->UpdateConnections();
        // 获取所有第二层所有共视帧
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        // 删除原有的二级链接
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        // 删除原有的和一级链接，两次删除后，剩下的就是更新后的即闭环引入的connecttion
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    // 构建优化构造器
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    // 增加闭环边
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    // 运行全局BA
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();

    mLastLoopKFid = mpCurrentKF->mnId;
}

// 根据修正后的pose，将闭环中的mappoint 更新current keyframe和相临keyframe中的mappoint
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        // 采用优化后的pose获取匹配融合的mappoints
        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        // 更新可替换的mappoint
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}


void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    bool idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                std::this_thread::sleep_for(std::chrono::microseconds(1000));
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
