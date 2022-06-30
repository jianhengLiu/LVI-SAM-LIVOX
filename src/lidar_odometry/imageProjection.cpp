#include "utility.h"
#include "lvi_sam/cloud_info.h"

#include <livox_ros_driver/CustomMsg.h>

// LIVOX AVIA
#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)
enum E_jump
{
    Nr_nor,
    Nr_zero,
    Nr_180,
    Nr_inf,
    Nr_blind
};
enum Feature
{
    Nor,
    Poss_Plane,
    Real_Plane,
    Edge_Jump,
    Edge_Plane,
    Wire,
    ZeroPoint
};
enum Surround
{
    Prev,
    Next
};
struct orgtype
{
    double  range;
    double  dista;
    double  angle[2];
    double  intersect;
    E_jump  edj[2];
    Feature ftype;
    orgtype()
    {
        range     = 0;
        edj[Prev] = Nr_nor;
        edj[Next] = Nr_nor;
        ftype     = Nor;
        intersect = 2;
    }
};

// Velodyne
struct PointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float    time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
                                  (float, x, x)(float, y, y)(float, z,
                                                             z)(float, intensity,
                                                                intensity)(uint16_t, ring,
                                                                           ring)(float, time, time))

// Ouster
// struct PointXYZIRT {
//     PCL_ADD_POINT4D;
//     float intensity;
//     uint32_t t;
//     uint16_t reflectivity;
//     uint8_t ring;
//     uint16_t noise;
//     uint32_t range;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// }EIGEN_ALIGN16;

// POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
//     (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
//     (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
//     (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
// )

const int queueLength = 500;

class ImageProjection : public ParamServer
{
private:
    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;

    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber              subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber                subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2>    cloudQueue;
    std::deque<livox_ros_driver::CustomMsg> livoxCloudQueue;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];  // 依据陀螺仪前推结果，没有考虑bias和noise
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int             imuPointerCur;  //点云对应imu的索引
    bool            firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int     deskewFlag;
    cv::Mat rangeMat;

    bool  odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lvi_sam::cloud_info cloudInfo;
    double              timeScanCur;   // 次次新scan的时间戳
    double              timeScanNext;  // 最新scan时间戳
    std_msgs::Header    cloudHeader;

    vector<int> columnIdnCountVec;  // for livox

public:
    ImageProjection() : deskewFlag(0)
    {
        ROS_INFO("\033[1;32m Subscribe to IMU Topic: %s.\033[0m", imuTopic.c_str());
        subImu  = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this,
                                                ros::TransportHints().tcpNoDelay());
        subOdom = nh.subscribe<nav_msgs::Odometry>(
            PROJECT_NAME + "/vins/odometry/imu_propagate_ros", 2000,
            &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        switch (lidarType)
        {
            ROS_INFO("\033[1;32m Lidar Type: %s.\033[0m", lidarType == VELO16 ? "VELO16" : "AVIA");
            ROS_INFO("\033[1;32m Subscribe to Lidar Topic: %s.\033[0m", pointCloudTopic.c_str());
            case VELO16:
                subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(
                    pointCloudTopic, 5, &ImageProjection::velo16CloudHandler, this,
                    ros::TransportHints().tcpNoDelay());
                break;
            case AVIA:
                subLaserCloud = nh.subscribe<livox_ros_driver::CustomMsg>(
                    pointCloudTopic, 5, &ImageProjection::aviaCloudHandler, this,
                    ros::TransportHints().tcpNoDelay());
                break;
            default:
                printf("Lidar type is wrong.\n");
                exit(0);
                break;
        }

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2>(
            PROJECT_NAME + "/lidar/deskew/cloud_deskewed", 5);
        pubLaserCloudInfo =
            nh.advertise<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/deskew/cloud_info", 5);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN * Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur  = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        columnIdnCountVec.assign(N_SCAN, 0);
    }

    ~ImageProjection() {}

    void imuHandler(const sensor_msgs::Imu::ConstPtr &imuMsg)
    {
        sensor_msgs::Imu            thisImu = imuConverter(*imuMsg);
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);
        // cout << "thisImu.header.stamp.toSec() = " << to_string(thisImu.header.stamp.toSec())
        //      << endl;

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << ", y: " <<
        // thisImu.linear_acceleration.y
        //      << ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << ", y: " << thisImu.angular_velocity.y
        //      << ", z: " << thisImu.angular_velocity.z << endl;
        // double         imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl
        //      << endl;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr &odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void velo16CloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        // 1 检查队列里面的点云数量是否满足要求 并做一些前置操作
        if (!cachePointCloud(laserCloudMsg))
            return;

        // 2 储存最新和次次新scan间的IMU和VO信息
        if (!deskewInfo())
            return;

        // 3 获取雷达深度图；点云去畸变
        projectPointCloud();

        // 4 点云提取
        cloudExtraction();

        publishClouds();

        resetParameters();
    }

    void aviaCloudHandler(const livox_ros_driver::CustomMsg::ConstPtr &laserCloudMsg)
    {
        // 1 检查队列里面的点云数量是否满足要求 并做一些前置操作
        if (!cachePointCloud(laserCloudMsg))
            return;

        // 2 储存最新和次次新scan间的IMU和VO信息
        if (!deskewInfo())
            return;

        // 3 获取雷达深度图；点云去畸变
        projectPointCloud();

        // 4 点云提取
        cloudExtraction();

        publishClouds();

        resetParameters();
    }

    //队列里面的点云数量是否满足要求
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        sensor_msgs::PointCloud2 currentCloudMsg = cloudQueue.front();
        cloudQueue.pop_front();

        cloudHeader  = currentCloudMsg.header;
        timeScanCur  = cloudHeader.stamp.toSec();
        timeScanNext = cloudQueue.front().header.stamp.toSec();

        // convert cloud
        pcl::fromROSMsg(currentCloudMsg, *laserCloudIn);

        // check dense flag
        if (laserCloudIn->is_dense == false)  ///表示点云里面没有去除无效点（NaN）
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        // 检查 点云是否包含ring通道(线数)
        // 该部分主要用来计算rowIdn
        // sensor_msgs::PointCloud2.fields:
        // https://blog.csdn.net/qq_45954434/article/details/116179169
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point "
                          "cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        // yaml文件中 timeField: "time"    # point timestamp field, Velodyne - "time", Ouster - "t"
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                //表示当前具有每个点时间戳信息
                if (currentCloudMsg.fields[i].name == timeField)
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system "
                         "will drift significantly!");
        }

        return true;
    }

    //队列里面的点云数量是否满足要求
    bool cachePointCloud(const livox_ros_driver::CustomMsg::ConstPtr &laserCloudMsg)
    {
        // cache point cloud
        livoxCloudQueue.push_back(*laserCloudMsg);
        if (livoxCloudQueue.size() <= 2)
            return false;

        livox_ros_driver::CustomMsg currentCloudMsg = livoxCloudQueue.front();
        livoxCloudQueue.pop_front();

        cloudHeader  = currentCloudMsg.header;
        timeScanCur  = cloudHeader.stamp.toSec();
        timeScanNext = livoxCloudQueue.front().header.stamp.toSec();
        // timeScanNext =
        //     timeScanCur + 1e-9 * static_cast<double>(currentCloudMsg.points.back().offset_time);
        // ;
        for (int i = 0; i < currentCloudMsg.point_num; i++)
        {
            if ((currentCloudMsg.points[i].line < N_SCAN) &&
                (!IS_VALID(currentCloudMsg.points[i].x)) &&
                (!IS_VALID(currentCloudMsg.points[i].y)) &&
                (!IS_VALID(currentCloudMsg.points[i].z)) && currentCloudMsg.points[i].x > 0.7)
            {
                // https://github.com/Livox-SDK/Livox-SDK/wiki/Livox-SDK-Communication-Protocol
                // See [3.4 Tag Information]
                if ((currentCloudMsg.points[i].x > 2.0) &&
                    (((currentCloudMsg.points[i].tag & 0x03) != 0x00) ||
                     ((currentCloudMsg.points[i].tag & 0x0C) != 0x00)))
                {
                    // Remove the bad quality points
                    continue;
                }
                PointXYZIRT point;
                point.x         = currentCloudMsg.points[i].x;
                point.y         = currentCloudMsg.points[i].y;
                point.z         = currentCloudMsg.points[i].z;
                point.intensity = currentCloudMsg.points[i].reflectivity;
                point.ring      = currentCloudMsg.points[i].line;
                point.time      = currentCloudMsg.points[i].offset_time;
                laserCloudIn->push_back(point);
            }
        }

        return true;
    }

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur ||
            imuQueue.back().header.stamp.toSec() < timeScanNext)
        {
            // cout << "imuQueue.front().header.stamp.toSec()="
            //      << to_string(imuQueue.front().header.stamp.toSec()) << endl;
            // cout << "imuQueue.back().header.stamp.toSec()="
            //      << to_string(imuQueue.back().header.stamp.toSec()) << endl;
            // cout << "timeScanCur=" << to_string(timeScanCur) << endl;
            // cout << "timeScanNext=" << to_string(timeScanNext) << endl;
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }
        // 2.1 储存lidar间IMU信息
        // TODO: 这里假设了lidar频率为10hz
        imuDeskewInfo();

        // 2.1 储存lidar间VO信息
        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() <
                timeScanCur - 0.01)  // 0.01s 这个时间怎么定的？因为雷达输出10hz吗？
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg     = imuQueue[i];
            double           currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                //将message里面的IMU消息转为tf类型的数据
                //前者只是个float的类型的结构体 后者则是一个类 封装了很多函数
                //将imu的朝向赋值给点云 ，如果非九轴imu在此处会不准：通过磁力计获得朝向
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit,
                              &cloudInfo.imuYawInit);

            if (currentImuTime > timeScanNext + 0.01)
                break;

            if (imuPointerCur == 0)
            {
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double timeDiff        = currentImuTime - imuTime[imuPointerCur - 1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        // TODO： 为了得到rpy有必要先转成tf吗，后面也没有用到tf，更多就是一个转换的介质
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        // 将点云信息中的位姿设置为视觉里程计的位姿
        cloudInfo.odomX       = startOdomMsg.pose.pose.position.x;
        cloudInfo.odomY       = startOdomMsg.pose.pose.position.y;
        cloudInfo.odomZ       = startOdomMsg.pose.pose.position.z;
        cloudInfo.odomRoll    = roll;
        cloudInfo.odomPitch   = pitch;
        cloudInfo.odomYaw     = yaw;
        cloudInfo.odomResetId = (int)round(startOdomMsg.pose.covariance[0]);

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;
        // 如果视觉里程计队里末尾值的时间戳小于两帧lidar时间戳 说明视觉里程计频率过低
        if (odomQueue.back().header.stamp.toSec() < timeScanNext)
            return;

        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanNext)
                continue;
            else
                break;
        }

        // 在visual_estimator/utility/visualization.cpp中发布给LIO视觉里程计信息的最后一项为failureCount
        // 初始值设为-1，每次clearState都会导致++failureCount
        // 如果前后的failureCount不一致说明在Lidar当前帧内视觉里程计至少重启了一次，跟踪失败，那么值就不准确了
        // 因此不对当前的视觉里程计去畸变
        if (int(round(startOdomMsg.pose.covariance[0])) !=
            int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(
            startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y,
            startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd =
            pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y,
                                   endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        // 依据VO获得最新scsan和次次新scan的相对位姿
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre,
                                          pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    // 找到一点最匹配的imu数据,把对应位姿赋给该点
    // 这里主要考虑问题,若一帧点云先到,而imu最新数据直到该点云帧扫描一半(不全)的情况
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0;
        *rotYCur = 0;
        *rotZCur = 0;

        int imuPointerFront = 0;
        //找到该点时间戳后最近的一个imu时间戳
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        // IMU最新数据老于该点or没有imu数据
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            //此时该点的姿态等于最新IMU的姿态
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        }
        else
        {
            // 插值得到该点的位姿
            // 时间轴: imuPointerBack < imuPointerCur < imuPointerFront
            int    imuPointerBack = imuPointerFront - 1;
            double ratioFront     = (pointTime - imuTime[imuPointerBack]) /
                                (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) /
                               (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    // 3.1.2 对位置信息进行去畸变
    // odomIncre依据VO得到，估计是VO输出结果不连续，导致效果不佳or运动不激烈不去畸变效果也可以
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0;
        *posYCur = 0;
        *posZCur = 0;

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanNext - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    // 3.1对点云中每个点进行去畸变
    //  由于载体存在移动，扫描一圈后事实上并不会形成圆形，但此时的xyz是相对于当前时刻的lidar，
    //  因为都属于同一帧所以需要转到最开始的lidar位置，也就是说对xyzrpy要加上一个位移和旋转
    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        //点云起始时间戳+相对于第一帧的时间戳
        double pointTime = timeScanCur + relTime;

        // 3.1.1对点云位置去畸变 实际上计算的相对于世界坐标系原点的变化
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        // 把第一点(start)的位姿作为lidar坐标系原点
        // 每当有新的一帧点云进来： firstPointFlag = true
        if (firstPointFlag == true)
        {
            // T_l(idar)s(tart) -> T_s(tart)l(idar)
            transStartInverse =
                (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur))
                    .inverse();
            firstPointFlag = false;
        }

        // transform points to start
        Eigen::Affine3f transFinal =
            pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        // T_sf(inal) = T_sl * T_lf
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        newPoint.x = transBt(0, 0) * point->x + transBt(0, 1) * point->y +
                     transBt(0, 2) * point->z + transBt(0, 3);
        newPoint.y = transBt(1, 0) * point->x + transBt(1, 1) * point->y +
                     transBt(1, 2) * point->z + transBt(1, 3);
        newPoint.z = transBt(2, 0) * point->x + transBt(2, 1) * point->y +
                     transBt(2, 2) * point->z + transBt(2, 3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    // 将深度信息投影到RangeImage上
    // 对点云进行去畸变； TODO: 为什么不先去畸变再投影？
    void projectPointCloud()
    {
        int cloudSize = (int)laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            // 一般雷达坐标系: x轴向前, y轴向左, z轴向上
            PointType thisPoint;
            thisPoint.x         = laserCloudIn->points[i].x;
            thisPoint.y         = laserCloudIn->points[i].y;
            thisPoint.z         = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            // 适配雷达线数，也可以设置N_SCAN来只使用部分线数据
            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            // 下采样
            if (rowIdn % downsampleRate != 0)
                continue;

            int columnIdn = -1;
            if (lidarType == VELO16)
            {
                // 计算该点水平角
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                // 计算分辨率(°)
                static float ang_res_x = 360.0 / float(Horizon_SCAN);
                // 获得该点对应图像columnId
                // 图像上的中心的左边对应y轴正方向,所以有负号
                columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;
            }
            else if (lidarType == AVIA)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn]++;
            }

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            float range = pointDistance(thisPoint);

            if (range < 1.0)
                continue;
            // 判断这个位置是否已经有点了，rangeMat初始化为FLT_MAX
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)  // FLOAT_MAX
                continue;

            // for the amsterdam dataset
            // if (range < 6.0 && rowIdn <= 7 && (columnIdn >= 1600 || columnIdn <= 200))
            //     continue;
            // if (thisPoint.z < -2.0)
            //     continue;

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // 3.1 对点云去畸变
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);  // Velodyne
            // thisPoint = deskewPoint(&thisPoint, (float)laserCloudIn->points[i].t / 1000000000.0);
            // // Ouster

            int index                = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    // 提取点云信息
    // 为什么不在projectPointCloud里面就完成这一步？rangeMat起到了什么作用？
    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.startRingIndex[i] = count - 1 + 5;  //最开始的五个不考虑

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i, j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i, j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count - 1 - 5;  //最末尾的五个不考虑
        }
    }

    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed =
            publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, "base_link");
        pubLaserCloudInfo.publish(cloudInfo); // TODO: ??? 怎么cloudInfo里面也有完整的点云信息？
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar");

    ImageProjection IP;

    ROS_INFO("\033[1;32m----> Lidar Cloud Deskew Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();

    return 0;
}