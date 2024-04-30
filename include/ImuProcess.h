#ifndef _IMU_PROCESS_H_
#define _IMU_PROCESS_H_

#include <Eigen/Core>
#include <memory>
struct ImuData
{
    double timestamp;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyro;
    ImuData(double timestamp, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyro) : timestamp(timestamp), acc(acc), gyro(gyro) {}
};

typedef std::shared_ptr<ImuData> ImuDataPtr;

struct State
{
    double timestamp;
    Eigen::Vector3d Pos;
    Eigen::Vector3d Vel;
    Eigen::Matrix3d Rot;
    ImuDataPtr imu_data_ptr;
    State() : Pos(Eigen::Vector3d::Zero()), Vel(Eigen::Vector3d::Zero()), Rot(Eigen::Matrix3d::Identity()), imu_data_ptr(nullptr) {}
    void reset(){
        Pos = Eigen::Vector3d::Zero();
        Vel = Eigen::Vector3d::Zero();
        Rot = Eigen::Matrix3d::Identity();
        imu_data_ptr = nullptr;
    }
};

class ImuProcessor
{

public:
    ImuProcessor() = default;
    State Predict(const State &last_state, const ImuDataPtr &cur_imu);
};

#endif