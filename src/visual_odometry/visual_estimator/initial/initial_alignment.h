#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"
#include "../parameters.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
public:
    ImageFrame(){};
    ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &_points,
               const vector<float> &_lidar_initialization_info, double _t)
        : t{_t}, is_key_frame{false}, reset_id{-1}, gravity{9.805}
    {
        points = _points;

        // reset id in case lidar odometry relocate
        reset_id = (int)round(_lidar_initialization_info[0]);
        // Pose
        T.x() = _lidar_initialization_info[1];
        T.y() = _lidar_initialization_info[2];
        T.z() = _lidar_initialization_info[3];
        // Rotation
        Eigen::Quaterniond Q =
            Eigen::Quaterniond(_lidar_initialization_info[7], _lidar_initialization_info[4],
                               _lidar_initialization_info[5], _lidar_initialization_info[6]);
        R = Q.normalized().toRotationMatrix();
        // Velocity
        V.x() = _lidar_initialization_info[8];
        V.y() = _lidar_initialization_info[9];
        V.z() = _lidar_initialization_info[10];
        // Acceleration bias
        Ba.x() = _lidar_initialization_info[11];
        Ba.y() = _lidar_initialization_info[12];
        Ba.z() = _lidar_initialization_info[13];
        // Gyroscope bias
        Bg.x() = _lidar_initialization_info[14];
        Bg.y() = _lidar_initialization_info[15];
        Bg.z() = _lidar_initialization_info[16];
        // Gravity
        gravity = _lidar_initialization_info[17];
    };

    map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> points;
    double                                                   t;

    IntegrationBase *pre_integration;
    bool             is_key_frame;

    // Lidar odometry info
    int      reset_id;
    Vector3d T;
    Matrix3d R;
    Vector3d V;
    Vector3d Ba;
    Vector3d Bg;
    double   gravity;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g,
                        VectorXd &x);

class odometryRegister
{
public:
    ros::NodeHandle    n;
    tf::Quaternion     q_lidar_to_imu;
    Eigen::Quaterniond q_lidar_to_imu_eigen;

    ros::Publisher pub_latest_odometry;

    odometryRegister(ros::NodeHandle n_in) : n(n_in)
    {
        // TODO: 修改为读配置文件决定外参
        // 怎么两个参数还是不一样的？
        //  1  0  0
        //  0 -1  0
        //  0  0 -1
        // q_lidar_to_cam = tf::Quaternion(0, 1, 0, 0); // rotate orientation // mark: camera - lidar
        // -1  0  0
        //  0 -1  0
        //  0  0  1
        // q_lidar_to_cam_eigen = Eigen::Quaterniond(0, 0, 0, 1); // rotate position by pi, (w, x, y, z) // mark: camera - lidar
        // // pub_latest_odometry = n.advertise<nav_msgs::Odometry>("odometry/test", 1000);
        // modified:
        q_lidar_to_imu       = tf::createQuaternionFromRPY(L_I_RX, L_I_RY, L_I_RZ);
        q_lidar_to_imu_eigen = Eigen::Quaterniond(
            q_lidar_to_imu.w(), q_lidar_to_imu.x(), q_lidar_to_imu.y(),
            q_lidar_to_imu.z());  // rotate position by pi, (w, x, y, z) // mark: camera - lidar
    }

    // convert odometry from ROS Lidar frame to VINS camera frame
    // DONE: ??? odomQueue 对应/odometry/imu 看起来是imu frame: 确实是lidar frame，可能是想表达imu频率的odometry
    vector<float> getOdometry(deque<nav_msgs::Odometry> &odomQueue, double img_time)
    {
        vector<float> odometry_channel;
        odometry_channel.resize(18, -1);  // reset id(1), P(3), Q(4), V(3), Ba(3), Bg(3), gravity(1)

        nav_msgs::Odometry odomCur;

        // pop old odometry msg
        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < img_time - 0.05)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
        {
            // ROS_INFO("No odometry msg!");
            // cout << "No odometry msg!" << endl;
            return odometry_channel;
        }

        // find the odometry time that is the closest to image time
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            odomCur = odomQueue[i];

            if (odomCur.header.stamp.toSec() < img_time - 0.002)  // 500Hz imu
                continue;
            else
                break;
        }

        // time stamp difference still too large
        if (abs(odomCur.header.stamp.toSec() - img_time) > 0.05)
        {
            return odometry_channel;
        }

        // modified:
        odometry_channel[0]  = odomCur.pose.covariance[0];
        odometry_channel[1]  = odomCur.pose.pose.position.x;
        odometry_channel[2]  = odomCur.pose.pose.position.y;
        odometry_channel[3]  = odomCur.pose.pose.position.z;
        odometry_channel[4]  = odomCur.pose.pose.orientation.x;
        odometry_channel[5]  = odomCur.pose.pose.orientation.y;
        odometry_channel[6]  = odomCur.pose.pose.orientation.z;
        odometry_channel[7]  = odomCur.pose.pose.orientation.w;
        odometry_channel[8]  = odomCur.twist.twist.linear.x;
        odometry_channel[9]  = odomCur.twist.twist.linear.y;
        odometry_channel[10] = odomCur.twist.twist.linear.z;
        odometry_channel[11] = odomCur.pose.covariance[1];
        odometry_channel[12] = odomCur.pose.covariance[2];
        odometry_channel[13] = odomCur.pose.covariance[3];
        odometry_channel[14] = odomCur.pose.covariance[4];
        odometry_channel[15] = odomCur.pose.covariance[5];
        odometry_channel[16] = odomCur.pose.covariance[6];
        odometry_channel[17] = odomCur.pose.covariance[7];

        return odometry_channel;
    }
};
