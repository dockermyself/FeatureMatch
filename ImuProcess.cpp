

#include <Eigen/Dense>
#include <iostream>
#include "ImuProcess.h"

State ImuProcessor::Predict(const State &last_state, const ImuDataPtr &cur_imu)
{
    State update_state;
    const ImuDataPtr& last_imu = last_state.imu_data_ptr;
    if(last_imu == nullptr)
    {
        update_state.timestamp = cur_imu->timestamp;
        update_state.imu_data_ptr = cur_imu;
        return update_state;
    }
    // Time.
    const double delta_t = cur_imu->timestamp - last_imu->timestamp;
    const double delta_t2 = delta_t * delta_t;
    
    // Acc and gyro.
    const Eigen::Vector3d acc_unbias = 0.5 * (last_imu->acc + cur_imu->acc);
    const Eigen::Vector3d gyro_unbias = 0.5 * (last_imu->gyro + cur_imu->gyro);

    // ENU坐标系下的加速度
    Eigen::Vector3d acc = last_state.Rot * acc_unbias + Eigen::Vector3d(0, 0, -9.8);
    // printf("acc: %f, %f, %f\n", acc(0), acc(1), acc(2));
    // Normal state.
    update_state.Pos = last_state.Pos + last_state.Vel * delta_t + 0.5 * acc * delta_t2;
    update_state.Vel = last_state.Vel + acc * delta_t;
    const Eigen::Vector3d delta_angle_axis = gyro_unbias * delta_t;
    if (delta_angle_axis.norm() > 1e-12)
    {
        update_state.Rot = last_state.Rot * Eigen::AngleAxisd(delta_angle_axis.norm(), delta_angle_axis.normalized()).toRotationMatrix();
    }

    // Time and imu.
    update_state.timestamp = cur_imu->timestamp;
    update_state.imu_data_ptr = cur_imu;

    return update_state;
}