#include "Frame.h"

Camera Frame::_Camera(416.9615, 416.4640, 330.4165, 233.7055, -0.0184, 0.0461, 0.0005304, -0.0004292, 640, 480, 0.25, 4.0);
int Frame::_FrameIdConter = 0;

void Frame::Match(const std::vector<MapPoint> &map_points, const Eigen::Matrix3d &R, const Eigen::Vector3d &T,
                  MatchPointContainer &matchs)
{
    // find the keypoint area radius = 40 (1米距离，横向10cm)
    const float radius = 0.075 * _Camera.focal_length();
    printf("radius: %f\n", radius);

    for (auto &map_point : map_points)
    {
        // Pw -> Pc
        const Eigen::Vector3d &Pw = map_point.Pos();
        // printf("Pw: %f, %f, %f\n", Pw.x(), Pw.y(), Pw.z());
        Eigen::Vector3d Pc = R * Pw + T;
        // Pc -> uv
        Eigen::Vector2d uv;
        if (!_Camera.project(Pc, uv))
        {
            continue;
        }

        std::vector<int> indices = GetFeaturesInArea(uv.x(), uv.y(), radius);
        // printf("indices size: %ld\n", indices.size());

        // 计算描述子相似度
        int best_dist = 256;
        int best_index = -1;
        for (auto &index : indices)
        {
            const cv::KeyPoint &kp = _keypoints[index];
            const cv::Mat &kp_descriptor = _descriptors.row(index);
            const cv::Mat &mp_descriptor = map_point.Descriptor();
            // 计算描述子距离
            int dist = DescriptorDistance(kp_descriptor, mp_descriptor);
            if (dist < best_dist)
            {
                best_dist = dist;
                best_index = index;
            }
        }

        if (best_dist < 50)
        {
            matchs.emplace_back(&map_point, &_keypoints[best_index]);
        }
    }
}

std::vector<int> Frame::GetFeaturesInArea(float u, float v, float radius)
{
    std::vector<int> indices;
    for (size_t i = 0; i < _keypoints.size(); i++)
    {
        if ((_keypoints[i].pt.x - u) * (_keypoints[i].pt.x - u) + (_keypoints[i].pt.y - v) * (_keypoints[i].pt.y - v) < radius * radius)
        {
            indices.push_back(i);
        }
    }
    return indices;
}