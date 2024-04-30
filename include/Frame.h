
#ifndef _FRAME_H_
#define _FRAME_H_
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Camera.h>
#include <memory>
#include "OrbExtractor.h"

class MapPoint;

class Frame
{

public:
    static Camera _Camera;
    static int _FrameIdConter;

private:
    Eigen::Matrix3d _R;
    Eigen::Vector3d _T;
    std::vector<cv::KeyPoint> _keypoints;
    cv::Mat _descriptors;
    const int _id;

public:
    using MatchPointContainer = std::vector<std::pair<const MapPoint *, const cv::KeyPoint *>>;
    Frame(std::vector<cv::KeyPoint> &&keypoints, const cv::Mat &descriptors)
        : _keypoints(std::move(keypoints)), _descriptors(descriptors), _id(_FrameIdConter++) {}

    const std::vector<cv::KeyPoint> &KeyPoints() const { return _keypoints; }
    const cv::Mat &Descriptors() const { return _descriptors; }
    int Id() const { return _id; }
    void Match(const std::vector<MapPoint> &map_points, const Eigen::Matrix3d &R, const Eigen::Vector3d &T,
               MatchPointContainer &matchs);
    std::vector<int> GetFeaturesInArea(float u, float v, float radius);

    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        for (int i = 0; i < 8; i++, pa++, pb++)
        {
            unsigned int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

    void SetPose(const Eigen::Matrix3d &R, const Eigen::Vector3d &T)
    {
        _R = R;
        _T = T;
    }
    Eigen::Matrix3d Rotation() const { return _R; }
    Eigen::Vector3d Translation() const { return _T; }
};

class MapPoint
{

    int _kpid;                                          // 特征点id
    double _lambda;                                     // 逆深度
    Eigen::Vector3d _pos;                               // 位置
    std::vector<std::weak_ptr<Frame>> _observed_frames; // 观测到该地图点的帧
    cv::Mat _descriptor;                                // 描述子

public:
    MapPoint(int kpid, const std::shared_ptr<Frame> &observed_frame, double lambda)
        : _kpid(kpid), _lambda(lambda)
    {
        const cv::KeyPoint &kp = observed_frame->KeyPoints()[kpid];
        _pos = Frame::_Camera.pixel2cam(kp.pt, lambda);
        _observed_frames.push_back(observed_frame);
        _descriptor = observed_frame->Descriptors().row(_kpid);
    }

    int Kpid() const { return _kpid; }

    double Lambda() const { return _lambda; }

    const cv::KeyPoint *KeyPoint() const
    {
        auto frame = _observed_frames.front().lock();
        if (!frame)
        {
            return nullptr;
        }

        return &frame->KeyPoints()[_kpid];
    }

    const Eigen::Vector3d &Pos() const { return _pos; }
    const cv::Mat &Descriptor() const { return _descriptor; }
};

#endif // _FRAME_H_
