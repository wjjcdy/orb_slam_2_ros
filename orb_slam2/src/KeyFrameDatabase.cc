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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{

// 构造函数，导入词袋数据
KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}

// 在词典文件中增加新的KF的单词，并记录单词对应的keyframe
void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

// 从词典中删除某kf所有单词
void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    // 遍历该keyframe中的每个单词
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        // 获取每个单词包含的keyframe list
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        // 如果存在与指定keyframe一致的删除
        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

// 清除整个用于闭环或重定位的词典集合
void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

// 检测与指定keyframe 有闭环关系的keyframe
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    // 获取所有与给定keyframe链接关系的keyframe，有链接关系是由连续时间关系的（即滑窗）
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    // 查找所有关键帧与当前帧存在共同特征点的，需要剔除链接关系的keyframe（即最近距离和时间内的keyframe）
    {
        unique_lock<mutex> lock(mMutex);

        // 遍历指定keyframe里所有单词
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            // 每个单词对应所有的keyframe list
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            // 遍历所有的keyframe
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                // 每个单词第一次搜索到
                if(pKFi->mnLoopQuery!=pKF->mnId)
                {
                    // 开始计数
                    pKFi->mnLoopWords=0;
                    // 存在共同单词的keyframe 不可以在关联的帧中
                    if(!spConnectedKeyFrames.count(pKFi))
                    {
                        // 记录发现为闭环的keyframe并记录id
                        pKFi->mnLoopQuery=pKF->mnId;
                        // 将有共同单词keyframe的放入 list中
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                // 记录每个其他kf与指定keyframe共同单词的个数
                pKFi->mnLoopWords++;
            }
        }
    }

    // 不存在共同单词的其他keyframe
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // 开辟用于匹配评分的容器
    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    // 获得具有共同单词的keyframe中，共同单词数最多的keyframe
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }

    // 获取最小共同的单词数
    int minCommonWords = maxCommonWords*0.8f;

    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    // 遍历所有共同单词的keyframe，仅共同单词数判断大于minCommonWords的keyframe
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        // 仅判断单词数大于阈值的
        if(pKFi->mnLoopWords>minCommonWords)
        {
            // 统计总共的个数
            nscores++;
            // 根据词典计算两帧的相似score
            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

            pKFi->mLoopScore = si;
            // 仅保存大于相似度阈值的keyframe
            if(si>=minScore)
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    // 无满足条件的keyframe
    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        // 共同单词的keyframe
        KeyFrame* pKFi = it->second;
        // 提取keyframe中关联度前10的keyframe
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi;
        // 遍历前10个相关的keyframe
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            // 与当前keyframe有闭环且共同单词数目大于阈值
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                // 累计匹配的分数
                accScore+=pKF2->mLoopScore;
                // 获取最大的匹配的分数和对应的keyframe
                if(pKF2->mLoopScore>bestScore)
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        // 记录累计的评分和对应最大的keyframe记录
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        // 获取最大累计的匹配评分
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    // 获取最小的累计评分
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    // 开启闭环候选帧的空间
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());
    // 遍历
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        // 仅判断累计评分大于一定阈值的
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            // 记录闭环候选帧，不重复
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
}

// 用于重定位的，查找与指定keyframe有共同单词的（即有关系）的keyframe队列
// 即用于重定位的候选kf
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        // 遍历currentframe每个单词
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            // 指定单词对应的所有kf list
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            // 遍历每个候选kf
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                // 未被遍历过，即仅添加一次
                if(pKFi->mnRelocQuery!=F->mnId)
                {
                    pKFi->mnRelocWords=0;
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                // 统计每个kf与current 共同单词的个数
                pKFi->mnRelocWords++;
            }
        }
    }
    // 若没有一个kf与指定frame有共同单词的话，则返回空
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    // 求出最多单词相同的个数
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    // 定义最小单词阈值
    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        // 仅对共同单词数大于阈值的kf进行判断和相似度评分
        if(pKFi->mnRelocWords>minCommonWords)
        {
            nscores++;
            // 根据词典的进行匹配评分
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    // 共同单词都不满足阈值，则返回空
    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    // 根据共视关系，再次进行累计评分
    // 遍历所有共同单词个数满足阈值的kf
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        // 获取此帧具有共视关系的前10个最佳的keyframe
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);
        // 该帧为最佳score的初始值
        float bestScore = it->first;
        // 初始评分累加值
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        // 遍历此帧的周围的10个最佳共视帧
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            // 表明没有一个与当前帧有共同单词的，可直接略过
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;
            // 累加评分
            accScore+=pKF2->mRelocScore;
            // 获取这10帧中最佳评分，并将最佳评分记录
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        // 将最佳评分和对应帧放入list中
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        // 获取最大累计评分
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    // 统计大于最高累计评分的0.75以上的kf，作为候选帧
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

// map serialization addition
template<class Archive>
void KeyFrameDatabase::serialize(Archive &ar, const unsigned int version)
{
    // don't save associated vocabulary, KFDB restore by created explicitly from a new ORBvocabulary instance
    // inverted file
    {
        unique_lock<mutex> lock_InvertedFile(mMutex);
        ar & mvInvertedFile;
    }
    // don't save mutex
}
template void KeyFrameDatabase::serialize(boost::archive::binary_iarchive&, const unsigned int);
template void KeyFrameDatabase::serialize(boost::archive::binary_oarchive&, const unsigned int);


} //namespace ORB_SLAM
