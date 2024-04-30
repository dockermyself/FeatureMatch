#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "OrbExtractor.h"
#include "SGBM.h"
#include "Frame.h"
#include "ImuProcess.h"
#include "ImuReader.h"

// 插值
bool Interpolate(const cv::Mat &disp, const cv::Point2f &pt, double &disp_value)
{
    cv::Point2i pt1(pt.x, pt.y);
    cv::Point2i pt2(pt.x + 1, pt.y + 1);
    // 插值计算视差
    double x = (pt.x - pt1.x) + (pt.y - pt1.y);
    ushort d1 = disp.at<ushort>(pt1.y, pt1.x);
    ushort d2 = disp.at<ushort>(pt1.y, pt2.x);
    // 低4位是小数部分
    float d1_f = (d1 & 0x0f) / 16.0 + (d1 >> 4);
    float d2_f = (d2 & 0x0f) / 16.0 + (d2 >> 4);

    if (d1 == 0 && d2 == 0)
    {
        return false;
    }
    else if (d1 == 0)
    {
        disp_value = d2_f;
        return true;
    }
    else if (d2 == 0)
    {
        disp_value = d1_f;
        return true;
    }
    else
    {
        disp_value = d1_f * x / 2.0 + d2_f * (1 - x / 2.0);
        return true;
    }
}

// 实现IMU数据读取
const ImuDataPtr ReadImuData(ImuReader &imu_reader)
{
    ImuOriginData data;
    if (imu_reader.getData(data))
    {
        Eigen::Vector3d acc = {data.accel[0].value, data.accel[1].value, data.accel[2].value};
        Eigen::Vector3d gyro = {data.gyro[0].value, data.gyro[1].value, data.gyro[2].value};
        double timestamp = data.timestamp.value;
        return std::make_shared<ImuData>(timestamp, acc, gyro);
    }

    return nullptr;
}

int main()
{

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error opening camera" << std::endl;
        return -1;
    }
    int imageW = 640;
    int imageH = 480;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, imageW * 2);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, imageH);

    // MJPEG
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    // fps
    cap.set(cv::CAP_PROP_FPS, 10);

    ORB::FeatureExtractor orb(500, 1.25, 7, 20, 7);
    Stereo::Sgbm sgbm(imageW, imageH);
    cv::Mat image;
    std::vector<MapPoint> map_point_container;
    std::shared_ptr<Frame> last_frame_ptr;
    std::shared_ptr<Frame> current_frame_ptr;
    cv::Mat last_image;
    cv::Mat current_image;
    std::mutex state_mutex;
    State state;
    std::thread imu_process([&state, &state_mutex]
                            {
            // 启动IMU线程
            ImuReader imu_reader("/dev/imu", B921600);
            imu_reader.EnableReadThread();
            ImuProcessor imu_processor;
            while (true)
            {
                ImuDataPtr imu_data_ptr = ReadImuData(imu_reader);
                if(imu_data_ptr){
                    std::lock_guard<std::mutex> lock(state_mutex);
                    state = imu_processor.Predict(state,imu_data_ptr);
                }
            } });

    while (true)
    {
        cap >> image;
        if (image.empty())
        {
            std::cerr << "Error reading image" << std::endl;
            break;
        }
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        cv::Mat imageL, imageR;
        imageL = image(cv::Rect(0, 0, image.cols / 2, image.rows));
        imageR = image(cv::Rect(image.cols / 2, 0, image.cols / 2, image.rows));
        current_image = imageL;
        cv::Mat grayL, grayR;
        cv::cvtColor(imageL, grayL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imageR, grayR, cv::COLOR_BGR2GRAY);
        sgbm.rectify(grayL, grayR);
        cv::Mat downsample_grayL, downsample_grayR;
        cv::resize(grayL, downsample_grayL, cv::Size(grayL.cols / 2, grayL.rows / 2));
        cv::resize(grayR, downsample_grayR, cv::Size(grayR.cols / 2, grayR.rows / 2));

        sgbm.compute(downsample_grayL, downsample_grayR);
        const cv::Mat &disp_matrix = sgbm.get_disp();
        // show disp_matrix
        // int nmDisparities = sgbm.get_nmDisparities();
        // cv::Mat show_disp;
        // disp_matrix.convertTo(show_disp, CV_8U, 255 / (nmDisparities * 16.));
        // cv::imshow("disp", disp_matrix);
        // cv::waitKey(1);
        // continue;

        std::vector<cv::KeyPoint> keypointsL, keypointsR;
        cv::Mat descriptorsL, descriptorsR;
        std::vector<int> vLappingArea{0, 0};
        orb.excute(grayL, cv::noArray(), keypointsL, descriptorsL, vLappingArea);
        orb.excute(grayR, cv::noArray(), keypointsR, descriptorsR, vLappingArea);
        current_frame_ptr = std::make_shared<Frame>(std::move(keypointsL), descriptorsL);
        // 设置位姿
        state_mutex.lock();
        current_frame_ptr->SetPose(state.Rot, state.Pos);
        state_mutex.unlock();

        Frame::MatchPointContainer matchs;

        if (!map_point_container.empty())
        {
            // 匹配

            Eigen::Matrix3d R = last_frame_ptr->Rotation().transpose() * current_frame_ptr->Rotation();
            Eigen::Vector3d T = R * (current_frame_ptr->Translation() - last_frame_ptr->Translation());
            std::cout << "Translation: " << T.transpose() << std::endl;
            current_frame_ptr->Match(map_point_container, R, T, matchs);
            //reset 
            state_mutex.lock();
            state.reset();
            state_mutex.unlock();
        }
        printf("matchs size: %ld\n", matchs.size());
        if (!matchs.empty())
        {
            // 绘制匹配点
            cv::Mat img_matches;
            cv::hconcat(last_image, current_image, img_matches);
            for (auto &match : matchs)
            {
                const MapPoint *map_point = match.first;
                const cv::KeyPoint *kp1 = map_point->KeyPoint();
                const cv::KeyPoint *kp2 = match.second;
                cv::Point2f pt1 = kp1->pt;
                cv::Point2f pt2 = cv::Point2f(kp2->pt.x + imageW, kp2->pt.y);
                cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
                cv::circle(img_matches, pt1, 2, color, 2);
                cv::circle(img_matches, pt2, 2, color, 2);
                cv::line(img_matches, pt1, pt2, color, 1);
            }
            cv::imshow("matches", img_matches);
            cv::waitKey(0);
        }

        map_point_container.clear();

        // 获取keypointsL的深度值
        for (size_t i = 0; i < current_frame_ptr->KeyPoints().size(); i++)
        {
            const cv::KeyPoint &kp = current_frame_ptr->KeyPoints()[i];
            const cv::Point2f pt = kp.pt * 0.5f; // 降采样
            double disp = 0.0;
            if (!Interpolate(disp_matrix, pt, disp))
            {
                continue;
            }

            Eigen::Vector3d world = sgbm.get_world(pt.x, pt.y, 2 * disp);

            if (world.z() < Frame::_Camera.min_depth_in_meter() * 1000 ||
                world.z() > Frame::_Camera.max_depth_in_meter() * 1000)
            {
                continue;
            }
            map_point_container.emplace_back(i, current_frame_ptr, 1000 / world.z());
        }
        last_image = current_image;
        last_frame_ptr = current_frame_ptr;
    }
    imu_process.join();
    cap.release();
    cv::destroyAllWindows();

    return 0;
}