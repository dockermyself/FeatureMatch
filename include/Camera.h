#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <opencv2/opencv.hpp>

struct Camera
{
    double fx, fy; // 焦距
    double cx, cy; // 光心
    double k1, k2; // 径向畸变
    double p1, p2; // 切向畸变

    const int image_w;
    const int image_h;
    const float min_depth;
    const float max_depth;

public:
    Camera(double fx, double fy, double cx, double cy,
           double k1, double k2, double p1, double p2,
           int image_w = 0, int image_h = 0, float min_depth = 0, float max_depth = 0)
        : fx(fx), fy(fy), cx(cx), cy(cy), k1(k1), k2(k2), p1(p1), p2(p2),
          image_w(image_w), image_h(image_h), min_depth(min_depth), max_depth(max_depth) {}

    float min_depth_in_meter() const { return min_depth; }
    float max_depth_in_meter() const { return max_depth; }
    cv::Mat cameraMatrix() const
    {
        static cv::Mat m = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        return m;
    }
    cv::Mat distCoeffs() const
    {
        static cv::Mat m = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, 0);
        return m;
    }

    // 从像素坐标系到归一化坐标系
    Eigen::Vector2d pixel2cam(const cv::Point2d &p) const
    {
        return {(p.x - cx) / fx, (p.y - cy) / fy};
    }
    // lambda = 1/d
    Eigen::Vector3d pixel2cam(const cv::Point2d &p, double lambda) const
    {
        return {1 / lambda * (p.x - cx) / fx, 1 / lambda * (p.y - cy) / fy, 1 / lambda};
    }
    // 从归一化坐标系到像素坐标系
    void project(const Eigen::Vector2d &p, Eigen::Vector2d &uv) const
    {
        uv.x() = p.x() * fx + cx;
        uv.y() = p.y() * fy + cy;
    }

    bool project(const Eigen::Vector3d &p, Eigen::Vector2d &uv) const
    {
        if (p(2) < min_depth || p(2) > max_depth)
            return false;

        uv.x() = fx * p.x() / p.z() + cx;
        uv.y() = fy * p.y() / p.z() + cy;

        if (uv.x() < 0 || uv.x() >= image_w || uv.y() < 0 || uv.y() >= image_h)
            return false;

        return true;
    }

    double focal_length() const { return (fx + fy) / 2; }
};

#endif // _CAMERA_H_