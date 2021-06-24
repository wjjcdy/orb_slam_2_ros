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

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{

// 初始化，选择匹配时，采用基础矩阵还是单应矩阵
// 单应矩阵H适用于,特征点主要在一平面内
// 基础矩阵适用于，特征点非一平面内

// 构建初始化函数，目的是设置参考帧
Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    // 设置参考关键帧
    mK = ReferenceFrame.mK.clone();

    // 去畸变后的所有特征点
    mvKeys1 = ReferenceFrame.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
}

// 初始化
/*
currentframe： 当前帧
vMatches12： 两帧特征匹配结果，即第1帧中的特征点在第二帧对应的index
R21： 输出到当前帧的旋转矩阵
t21： 输出当前帧的平移矩阵
vP3D ：
*/
bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    // 当前帧的去畸变后的特征点
    mvKeys2 = CurrentFrame.mvKeysUn;

    // 将匹配和索引重新封装新的类型
    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());
    // 记录匹配点有效还是无效， 
    // ？？两帧检测的个数并一样，匹配的会比单帧中所有点少，这里为何如此开辟空间
    mvbMatched1.resize(mvKeys1.size());
    // 遍历特征点匹配结果
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        // 如果有效，则将序号和value重新封装
        if(vMatches12[i]>=0)
        {
            // first 为第一帧中特征点index， second 为对应第二帧中index
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;
        }
        else
            mvbMatched1[i]=false;
    }

    // 有效匹配个数
    const int N = mvMatches12.size();

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    // 产生mMaxIterations组，每组包含8 point 的集合
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0);

    // 迭代mMaxIterations
    // 其主要目的就是 产生迭代mMaxIterations组数据，每组数据包含8个index，所有index不重复
    // 采用8点法，即至少8对点可求解出本质矩阵E，4对点可求单应矩阵H
    for(int it=0; it<mMaxIterations; it++)
    {
        // 复制所有特征点对 
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            // 从有效index中产生随机数
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];
            // 记录此次迭代的队列id
            mvSets[it][j] = idx;
            // 将最后一个中的value替换已提取的位置的value，并删除最后一个
            // 实际上就是取出一个，则删除这个（vector中直接从当前位置删除）
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    float SH, SF;
    cv::Mat H, F;

    // 开辟两个线程分别计算单应矩阵H和基础矩阵F
    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished
    threadH.join();
    threadF.join();

    // Compute ratio of scores
    // 计算评分比例
    float RH = SH/(SH+SF);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    // 选择最佳矩阵类型，然后根据H或F求出R和t
    if(RH>0.40)
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    else //if(pF_HF>0.6)
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    return false;
}

// 计算H单应矩阵,并获取评分
/*
都是输出
vbMatchesInliers： 记录误差过大的野点和非野点
score： 评分
H21： 输出单应矩阵
*/
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    // 特征点对
    const int N = mvMatches12.size();

    // Normalize coordinates
    // 两帧的特征点归一化
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    // 根据迭代次数迭代，并求出score最高的一组8点
    // 采用RANSAC 即随机采样一致性求解方程
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[it][j];
            // 分别两帧对应的特征点对，共8对
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
        // 根据8对计算H
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
        // 还原真实的单应矩阵, 从1到2的单应矩阵
        H21i = T2inv*Hn*T1;
        // 从2到1的单应矩阵
        H12i = H21i.inv();

        // 根据计算的H矩阵计算得分
        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

        // 获得迭代中最大得分的单应矩阵和评分，并记录有效的匹配点对（即剔除误差过大的点对）
        if(currentScore>score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

// 计算基础矩阵F
/*
与H一致，均是输出
*/
void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    // 所有特征点归一化
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        // 提取8对对应特征点对
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
        //根据8个点对，求取F矩阵
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);
        // 还原真实的基础矩阵F
        F21i = T2t*Fn*T1;

        // 对F进行验证，并评分
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

// 根据对应对，求取H单应矩阵
// 具体解法可看视觉14讲中7.3节
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2*N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        // 对应点对
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;

        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        // 构建矩阵
        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;

    // 采用SVD求解
    // A = u* w * v
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    // 由于为齐次方程，故最优解为vt的最后一行值，即第9行
    // 1，9 变成3，3
    // 原来是1通道1个数，共1行，共9个数
    return vt.row(8).reshape(0, 3);      // 通道不变， 变成3行
}

// 根据八点对，计算F（）
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F);
    // 结算公式可参考视觉14讲中的7.3节
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;
    // SVD
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    // vt最后一行，改成3*3矩阵，即为F矩阵
    cv::Mat Fpre = vt.row(8).reshape(0, 3);
    // 将F SVD分解，F为3*3
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    // 根据E的特性，w=diag（g，g，0），即第三奇异值应为0，故强行赋为0
    w.at<float>(2)=0;

    // cv::Mat::diag(w)生成对角矩阵？？？？ w本来应该就是个对角矩阵
    return  u*cv::Mat::diag(w)*vt;
}

// 判断单应矩阵的准确性，采用一定评分机制
float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    const int N = mvMatches12.size();

    // 从1kf到2kf的3*3 单应矩阵
    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    // 从2kf到1kf的 3*3单应矩阵
    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 5.991;

    // 误差系数
    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        // 对应匹配对特征点坐标
        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2， 即kf2中的特征点咋kf1中的投影
        // 尺度
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        //kf2中的特征点在kf1中的投影
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        // 误差平方和
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        // 缩放，本质应该是归一化
        const float chiSquare1 = squareDist1*invSigmaSquare;

        // 根据投影误差判断是否超过阈值
        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // 检测kf1中的特征点在kf2中的投影误差
        // x1in2 = H21*x1

        // 尺度
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        // 投影坐标
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        // 标记误差过大的点对，即将剔除误差过大的
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

// 判断本质矩阵的准确性
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();
    // F矩阵
    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;
    // 阈值
    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];
        // 对应点对
        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // 在kf1中的点p在kf2中的投影的极线

        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;
        // 投影的极线与（u2，v2，1）内积理论上应该为0
        const float num2 = a2*u2+b2*v2+c2;

        // 误差
        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

// 从本质矩阵中提取旋转矩阵和平移信息
/*
vbMatchesInliers: 有效点对
F21： 基础矩阵
k: 相机内参
R21: 即从kf1到kf2的旋转
t21：
vP3D: kf2内的特征点在kf1下的投影
vbTriangulated： 特征点是否重投影成功的标志位
minParallax ： 设置的最小视差角余弦值参数，
minTriangulated：匹配点中H21重投影成功的个数如果小于此值，返回失败
*/
bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    // 统计内点的个数（即有效匹配对）
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    // 求出本质矩阵
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    // 从本质矩阵中，可恢复4组 R和t 解
    DecomposeE(E21,R1,R2,t);  
    // t存在两个解， 故组合共有4组解
    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    // 验证4组解，求出每组投影误差小于阈值，点对数最多的点
    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    // 求取个数最多的点对
    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    // 匹配满足阈值最小点对数 阈值
    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    // 这段目的实际是想证明 只有一个个数比较多，其他三个个数非常小，即有至少30%差距
    // nsimilar>1,则表明不能突出这个具有明显正确性
    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    // 满足两个条件，则证明此解，即初始化成功
    // 匹配ok的点对数超过阈值，此解明显优于两三组解
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    // 选择最优的那组解，作为最终解
    if(maxGood==nGood1)
    {
        // 两帧夹角得大于一定阈值，即初始化时，要求两帧有一定的平移
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

// 从单应矩阵中提取旋转矩阵和平移信息
/*
采用SFM方法
参数与 F解析一致
*/
bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

    float s = cv::determinant(U)*cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }


    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

// 采用三角测量方法，计算3d pose
// opencv 中有现成库
void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

// 特征点归一化
// 归一化后的坐标点集合均值为0（去中心话，即减去原始均值），方差为1（缩放）
void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    // 所有特征点质心坐标
    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    // 协方差累加和
    for(int i=0; i<N; i++)
    {
        // 将均值坐标作为原点坐标，获取归一化后的特征点坐标
        // 如此均值理论为0
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;
        // 协方差绝对值累计和
        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    // 平均方差
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    // 方差倒数
    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    // 根据方差将point缩放
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    // 输出归一化变换的矩阵
    // x = (x-meanx)*sX
    // y = (y-meanY)*sY
    // 故矩阵形式为 [sX 0   -meanX*sX] * [x]
    //            [0  sY  -meanY*sY]   [y]
    //            [0  0   0        ]   [1]               
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

// 验证R和t
/*
R, 旋转矩阵
t, 平移矩阵
vKeys1, 帧1的特征点
vKeys2, 帧2的特征点
vMatches12, orb对应点对index
vbMatchesInliers, 匹配点中有效匹配点（符合H或F的投影变换的）
K, 相机内参
vP3D, 三角化重投影成功的匹配点的3d点在相机1下的坐标
th2, 根据三角化重投影误差判断匹配点是否重投影成功的阈值
vector<bool> &vbGood, //特征点哪些三角化重投影成功
float &parallax)//三角化重投影成功匹配点的视差角
*/
int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    // kf1的 变换矩阵T1 3*4
    // 可将T1认为是原点（T1可认为世界坐标系重合），故为单位矩阵即K[I|0]
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    // 非齐次坐标，kf1的光心的世界坐标
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    // kf2的变换矩阵T2, T2则是从T1到T2的变换，由于T1为原点，故T2= K[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    // 考虑内参，即换算成相机坐标（从像素换算成相机坐标）
    P2 = K*P2;

    // 非齐次坐标， kf2的光心世界坐标
    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        // 非有效对，跳过
        if(!vbMatchesInliers[i])
            continue;

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;
        // 采用三角测量方法，计算每个点的3D POSE
        Triangulate(kp1,kp2,P1,P2,p3dC1);

        // 若 3d pose存在无效值，则记录投影失败
        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        // 计算该点到kf1的光心的向量，并求出dist
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        // 计算该点到kf2的光心的向量，并求出dist
        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        // 该特征点到达两帧相机光心的向量的 夹角余弦
        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        // 该点的在kf1的深度在为负，则无效
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;
        // 该点在kf2的深度在为负，则无效
        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        // 根据深度信息，推出在图1中的像素坐标
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        // 计算像素坐标和观测值的误差
        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        // 误差过大，则跳过
        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        // 同理，计算在图2中国像素误差
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;

        // 记录夹角余弦
        vCosParallax.push_back(cosParallax);
        // 记录kf1中特征点的3d pose
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        // 统计goo的 point的个数
        nGood++;

        // 标记good的index
        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    // 目的就是找一个较大的视角
    if(nGood>0)
    {
        // 从小到达排序
        sort(vCosParallax.begin(),vCosParallax.end());

        // 取第50个的value，如果总数小于50的，就取最后一个index
        size_t idx = min(50,int(vCosParallax.size()-1));
        // 推出视角
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    // 返回较好，即误差较小对应的点对数
    return nGood;
}

// 从本质矩阵中，提取R1和R2,以及t，
// 与视觉14讲公式不太一样？？？？
// 视觉14讲，恢复pose直接采用openv现成的库
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    // 将本质均值进行svd
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    // 绕z轴旋转90度
    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    // 绕z轴反转90度
    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
