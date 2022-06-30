#include "utility.h"
#include "lvi_sam/cloud_info.h"

struct smoothness_t
{
    float  value;
    size_t ind;
};

struct by_value
{
    bool operator()(smoothness_t const &left, smoothness_t const &right)
    {
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{
public:
    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    lvi_sam::cloud_info cloudInfo;
    std_msgs::Header    cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float                    *cloudCurvature;
    int                      *cloudNeighborPicked;
    int                      *cloudLabel;

    FeatureExtraction()
    {
        subLaserCloudInfo = nh.subscribe<lvi_sam::cloud_info>(
            PROJECT_NAME + "/lidar/deskew/cloud_info", 5, &FeatureExtraction::laserCloudInfoHandler,
            this, ros::TransportHints().tcpNoDelay());

        pubLaserCloudInfo =
            nh.advertise<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/feature/cloud_info", 5);
        // 下面两个只是用于可视化
        pubCornerPoints =
            nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_corner", 5);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>(
            PROJECT_NAME + "/lidar/feature/cloud_surface", 5);

        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN * Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize,
                                   odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature      = new float[N_SCAN * Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN * Horizon_SCAN];
        cloudLabel          = new int[N_SCAN * Horizon_SCAN];
    }

    void laserCloudInfoHandler(const lvi_sam::cloud_infoConstPtr &msgIn)
    {
        cloudInfo   = *msgIn;                                     // new cloud info
        cloudHeader = msgIn->header;                              // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud);  // new cloud for extraction

        // 2、计算点云中各点曲率，为提取特征点准备
        calculateSmoothness();

        // 3、屏蔽点云中被遮挡点或平行点
        markOccludedPoints();

        // 4、依据曲率，提取角特征和面特征
        extractFeatures();

        // 5、向后续节点发布提取到的角特征和面特征
        publishFeatureCloud();
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        // 根据曲率计算公式，所以起始5个点(序号0-4)和最后五个点，不计算曲率；
        for (int i = 5; i < cloudSize - 5; i++)
        {
            //曲率计算公式 = 当前点前5个点距离和 - 当前点距离*10 + 当前点后5个点距离和；
            float diffRange = cloudInfo.pointRange[i - 5] + cloudInfo.pointRange[i - 4] +
                              cloudInfo.pointRange[i - 3] + cloudInfo.pointRange[i - 2] +
                              cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i] * 10 +
                              cloudInfo.pointRange[i + 1] + cloudInfo.pointRange[i + 2] +
                              cloudInfo.pointRange[i + 3] + cloudInfo.pointRange[i + 4] +
                              cloudInfo.pointRange[i + 5];

            // 曲率采用距离的平方的形式
            cloudCurvature[i] =
                diffRange * diffRange;  // diffX * diffX + diffY * diffY + diffZ * diffZ;

            //  计算特征点的标记位
            // 1:表示不参与提取特征点，可能受遮挡等
            // 0:表示参与提取特征点
            // 默认为参与提取特征点
            cloudNeighborPicked[i] = 0;
            // 特征点的分类标记
            // 1：曲率比较大
            // 0:默认情况下，比较平坦点，现实世界也是以平面点为主；
            // 不满足1、-1的情况下，都是0比较平坦点
            // -1:平坦的点
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            // 该点索引，后面需对点曲率排序会造成混乱，这个时候索引的作用就体现出来了;
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind   = i;
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1   = cloudInfo.pointRange[i];
            float depth2   = cloudInfo.pointRange[i + 1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i + 1] - cloudInfo.pointColInd[i]));

            // 如果相邻两个点，其水平角度接近，则距离中心更远的点及其周围5个点视为被遮挡点；
            if (columnDiff < 10)
            {
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3)
                {
                    // 1:表示不参与提取特征点，可能受遮挡等
                    // 0:表示参与提取特征点
                    // 默认为参与提取特征点
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i]     = 1;
                }
                else if (depth2 - depth1 > 0.3)
                {
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            float diff1 = std::abs(float(cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i + 1] - cloudInfo.pointRange[i]));

            // 如果点和相邻两个点的距离差都很大，则可能是平行线束，视为被遮挡点；
            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        // 保证提取的特征整体分布较为均匀
        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            for (int j = 0; j < 6; j++)
            {
                //每个scan6等分
                int sp =
                    (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep =
                    (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) /
                        6 -
                    1;

                if (sp >= ep)
                    continue;

                // 对当前提取段的曲率进行排序
                // 按照曲率从小到大：by_value();
                // cloudSmoothness:
                // 点的index和曲率，排序按照曲率排序，尽管cloudSmoothness被打乱了，可以用index与曲率的关联找到,以及该点状态；
                std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    //利用cloudSmoothness中index可以关联到该点是否参与提取特征点的状态
                    int ind = cloudSmoothness[k].ind;
                    // 0：参与提取特征点，曲率大于阈值，则算1个合格特征点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        //每个scan分成6段，每段提取满足条件的曲率前20大的点作为特征点
                        if (largestPickedNum <= 20)
                        {
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        //避免特征点过于集中，将当前点周围5个点状态置为1，不参与特征提取
                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] -
                                                          cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] -
                                                          cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // 面点特征提取，相同道理，每段scan分成6段；
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        cloudLabel[ind]          = -1;
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] -
                                                          cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] -
                                                          cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // TODO: 为什么不放在前面的循环中就把点堆入？
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0)
                    {
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            //避免特征过大，造成计算时间过长，对面点进行降采样；
            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.startRingIndex.shrink_to_fit();
        cloudInfo.endRingIndex.clear();
        cloudInfo.endRingIndex.shrink_to_fit();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointColInd.shrink_to_fit();
        cloudInfo.pointRange.clear();
        cloudInfo.pointRange.shrink_to_fit();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner =
            publishCloud(&pubCornerPoints, cornerCloud, cloudHeader.stamp, "base_link");
        cloudInfo.cloud_surface =
            publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, "base_link");
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Lidar Feature Extraction Started.\033[0m");

    ros::spin();

    return 0;
}