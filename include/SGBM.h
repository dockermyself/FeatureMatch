#ifndef _SGBM_H_
#define _SGBM_H_
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Camera.h>

namespace Stereo
{
    /*
        IntrinsicMatrix:
        416.9615        0         0
            0       416.4640      0
        330.4165    233.7055    1.0000
        RadialDistortion -0.0184    0.0461
        TangentialDistortion 0.0005304   -0.0004292

    */
    const Camera _cameraL(416.9615, 416.4640, 330.4165, 233.7055, -0.0184, 0.0461, 0.0005304, -0.0004292);
    /*
        IntrinsicMatrix:
        415.7149         0          0
         0           415.5435       0
        318.5744     230.9292    1.0000
        RadialDistortion -0.0245    0.0601
        TangentialDistortion -0.0000   -0.0020

    */
    const Camera _cameraR(415.7149, 415.5435, 318.5744, 230.9292, -0.0245, 0.0601, 0.0000, -0.0020);

    /*
    _R是旋转矩阵，left camera到right camera的旋转矩阵
    1.0000   -0.0003   -0.0100
    0.0003    1.0000   -0.0011
    0.0100    0.0011    0.9999
    */
    const cv::Mat _R = (cv::Mat_<double>(3, 3) << 1.0000, -0.0003, -0.0100, 0.0003, 1.0000, -0.0011, 0.0100, 0.0011, 0.9999);
    /*
    _T是平移矩阵，left camera到right camera的平移矩阵
    -118.8206    0.0842   -1.1373
    */
    const cv::Mat _T = (cv::Mat_<double>(3, 1) << -118.8206, 0.0842, -1.1373);

    cv::Mat _mapLx, _mapLy, _mapRx, _mapRy;
    cv::Mat _Q;

    class Sgbm
    {

        int _nmDisparities = 0;
        cv::Ptr<cv::StereoSGBM> sgbm = cv::Ptr<cv::StereoSGBM>();
        cv::Mat _disp;

    public:
        Sgbm(int width, int height) : sgbm(cv::StereoSGBM::create())
        {
            int winSize = 15;
            _nmDisparities = ((width / 8) + 15) & -16; // 视差搜索范围
            sgbm->setPreFilterCap(31);                 // 预处理滤波器截断值
            sgbm->setBlockSize(winSize);               // SAD窗口大小
            sgbm->setP1(8 * winSize * winSize);        // 控制视差平滑度第一参数
            sgbm->setP2(32 * winSize * winSize);       // 控制视差平滑度第二参数
            sgbm->setMinDisparity(0);                  // 最小视差
            sgbm->setNumDisparities(_nmDisparities);   // 视差搜索范围
            sgbm->setUniquenessRatio(5);               // 视差唯一性百分比
            sgbm->setSpeckleWindowSize(100);           // 检查视差连通区域变化度的窗口大小
            sgbm->setSpeckleRange(32);                 // 视差变化阈值
            sgbm->setDisp12MaxDiff(1);                 // 左右视差图最大容许差异
            sgbm->setMode(cv::StereoSGBM::MODE_SGBM);  // 采用全尺寸双通道动态编程算法
            cv::Mat Rl, Rr, Pl, Pr;
            cv::Size image_size = cv::Size(width, height);
            cv::stereoRectify(_cameraL.cameraMatrix(), _cameraL.distCoeffs(),
                              _cameraR.cameraMatrix(), _cameraR.distCoeffs(),
                              image_size, _R, _T, Rl, Rr, Pl, Pr, _Q,
                              cv::CALIB_ZERO_DISPARITY, 0, image_size);
            cv::initUndistortRectifyMap(_cameraL.cameraMatrix(), _cameraL.distCoeffs(), Rl, Pl, image_size, CV_32FC1, _mapLx, _mapLy);
            cv::initUndistortRectifyMap(_cameraR.cameraMatrix(), _cameraR.distCoeffs(), Rr, Pr, image_size, CV_32FC1, _mapRx, _mapRy);
        }

        void compute(const cv::Mat &imageL, const cv::Mat &imageR)
        {
            cv::Mat disp(imageL.size(), CV_16S);
            sgbm->compute(imageL, imageR, _disp);
            // _disp = _disp.mul(downsample);
            // cv::reprojectImageTo3D(_disp, _world, _Q, true); // 生成三维点云
            // _world = _world.mul(16);                         // _world乘以16
        }

        void rectify(cv::Mat &imageL, cv::Mat &imageR)
        {
            cv::remap(imageL, imageL, _mapLx, _mapLy, cv::INTER_LINEAR);
            cv::remap(imageR, imageR, _mapRx, _mapRy, cv::INTER_LINEAR);
        }

        inline void rectify_show(const cv::Mat &frame)
        {
            cv::Mat imageL = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
            cv::Mat imageR = frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
            cv::remap(imageL, imageL, _mapLx, _mapLy, cv::INTER_LINEAR);
            cv::remap(imageR, imageR, _mapRx, _mapRy, cv::INTER_LINEAR);
            // 绘制线条
            for (int i = 0; i < frame.rows; i += 20)
            {
                cv::line(frame, cv::Point(0, i), cv::Point(frame.cols, i), cv::Scalar(0, 255, 0), 1);
            }
            cv::imshow("rectify", frame);
            cv::waitKey(0);
        }

        Eigen::Vector3d get_world(int x, int y, double d)
        {
            cv::Mat w = _Q * cv::Mat(cv::Vec4d(x, y, d, 1));
            return {w.at<double>(0) / w.at<double>(3),
                    w.at<double>(1) / w.at<double>(3),
                    w.at<double>(2) / w.at<double>(3)};
        }

        cv::Mat get_disp()
        {
            return _disp;
        }

        int get_nmDisparities()
        {
            return _nmDisparities;
        }
    };

}

#endif // _SGBM_H_