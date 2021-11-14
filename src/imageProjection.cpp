// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above groundScanInd notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.


//  - pc - - - - - - ;



#include "utility.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

class ImageProjection{
private:

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    
    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;           // 
    pcl::PointCloud<PointXYZIR>::Ptr laserCloudInRing;

    pcl::PointCloud<PointType>::Ptr fullCloud;              //  projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<PointType>::Ptr fullInfoCloud;          //  intensit  same as fullCloud, but with intensity - range

    pcl::PointCloud<PointType>::Ptr groundCloud;            // 
    pcl::PointCloud<PointType>::Ptr segmentedCloud;         //  
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;     //  
    pcl::PointCloud<PointType>::Ptr outlierCloud;           // 

    PointType nanPoint; // fill in fullCloud at each iteration

    char szSaveName[1024];    / size
    int SaveNameNum = 1;    / 

    cv::Mat rangeMat;   // range matrix for range image  
    cv::Mat rangeMatTemp;   // rangeMat 
    cv::Mat rangeMatDiff;
    cv::Mat rangeMat2st;
    cv::Mat labelMat;   // label matrix for segmentaiton marking  
    cv::Mat labelMat_Copy;      // Ma  
    cv::Mat labelMat_Hard;      // 
    cv::Mat groundMat;  // ground matrix for ground cloud marking  

    cv::Mat MatTemp;
    
    int labelCount;
    int labelCount_Hard;

    float startOrientation;
    float endOrientation;

    cloud_msgs::cloud_info segMsg; // info of segmented cloud
    std_msgs::Header cloudHeader;

    std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process


    uint16_t *allPushedIndX; // array for tracking points of a segmented object 
    uint16_t *allPushedIndY;



    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed 
    uint16_t *queueIndY;


    int Count_full = 0;

    int N_num_ground_full = 0;
    int N_num_usessd_full = 0;
    int H_num_usessd_full = 0;
    int C_num_usessd_full = 0;




public:
    ImageProjection():
        nh("~"){
        
        // ：pointCloudTopic   utility.h 
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler, this);   // callback

        // 
        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        // 
        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);


        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;    // 

        allocateMemory();
        resetParameters();
    }

    // 
    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudInRing.reset(new pcl::PointCloud<PointXYZIR>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);


        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];


        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];





    }

    // 
    void resetParameters(){
        laserCloudIn->clear();      
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        // rangeMatTemp = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));     // rangeMatTemp
        // rangeMatDiff = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));     // rangeMatTemp

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        rangeMat2st = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelMat_Copy = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelMat_Hard = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));

        MatTemp = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));

        labelCount = 1;
        labelCount_Hard = 1;


        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    ~ImageProjection(){}        // 

    // 
    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        cloudHeader = laserCloudMsg->header;
        // cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);

        // Remove Nan points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);

        // have "ring" channel in the cloud
        if (useCloudRing == true){
            pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing);
            if (laserCloudInRing->is_dense == false) {
                ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
                ros::shutdown();
            }  
        }
    }
    
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        // 1. Convert ros message to pcl point cloud
        // 1. ro PC 
        // RO PointClou PC PointCloud RO   nan ；
        copyPointCloud(laserCloudMsg); / pc 

        // 2. Start and end angle of a scan
        // 2. 
        // 360° cloud_msgs::cloud_info segMs   segMs     fullClou 
        findStartEndAngle(); //1  atan ; 2 

        // 3. Range image projection
        // 3. 3 
        //        Ma  pcl::PointCloud fullCloud/fullInfoClou  （fullCloud （fullInfoCloud  PointClou     PointClou    livo  。
        projectPointCloud(); / 16x1800  


        // rangeMatRemoval();  //  )
        // SaveNameNum++;

        // 4. Mark ground points
        // 4. 
        //  groundMa  ，labelMa -1 na   。
        groundRemoval(); /  1 
        

        labelHandler();


        // 5. Point cloud segmentation
        // 5. 
        //   patc  labelMa patc   ） patc segmentedClou  nod nod  segMs    ）。
        cloudSegmentation(); /  ;

        // 6. Publish all clouds
        // 6. 
        //  segMsg fullCloud/fullInfoCloud 3）  4）  5）  5 patc ）。
        publishCloud(); // 

        // 7. Reset parameters for next iteration
        // 7.  
        resetParameters();
    }


    // void rangeMatRemoval(){

    //     // rangeMatTemp = rangeMat;
    //     string SaveName;
    //     float rangeMatDiffMax = 0.0;

    //     // path name
    //     // sprintf(szSaveName, "/home/scale/catkin_ws/rangeviewDiff%03d.png", SaveNameNum);
    //     sprintf(szSaveName, "/home/scale/catkin_ws/rangeviewDiff%03d.xml", SaveNameNum);
    //     SaveName.assign( szSaveName );
    //     cout << SaveName << "";
    //     if(SaveNameNum%10 == 0){
    //         cout << endl;
    //     }


    //     //  rangeMat  rangeMatTemp 
    //     for(size_t x = 0; x < N_SCAN; ++x){
    //         for(size_t y = 0; y < Horizon_SCAN; ++y){
    //             // rangeMatDiff.at<float>(x, y) = fabs(rangeMat.at<float>(x, y) 
    //             //                                             - rangeMatTemp.at<float>(x, y));
    //             rangeMatDiff.at<float>(x, y) = rangeMat.at<float>(x, y);
                
    //             // 
    //             // if(rangeMatDiff.at<float>(x, y) > rangeMatDiffMax){
    //             //     rangeMatDiffMax = rangeMatDiff.at<float>(x, y);
    //             // }
    //             // cout << rangeMatDiff.at<float>(x, y) << "" << endl;
    //         }
    //     }

    //     //  0～255
    //     // for(size_t x = 0; x < N_SCAN; ++x){
    //     //     for(size_t y = 0; y < Horizon_SCAN; ++y){
    //     //         rangeMatDiff.at<float>(x, y) = round((255-0) * (rangeMatDiff.at<float>(x, y)-0) / (rangeMatDiffMax-0) + 0);
    //     //     }
    //     // }
        

    //     // cv::FileStorage fs("rangeMatDiff.xml", FileStorage::WRITE);
    //     FileStorage fs;
    //     fs.open(SaveName, FileStorage::WRITE);
	//     fs << "rangeMatDiff" << rangeMatDiff;
	//     fs.release();

    //     // imwrite(szSaveName, rangeMatDiff);   // 
    // }


    void findStartEndAngle(){

        //  360° 360    0~377  ro  360  360           2 270 
        // ？ 

        // start and end orientation of this cloud
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        // ？ atan2(  atan2( -π ~ +π   
        // ？ velodyn  

        segMsg.endOrientation   = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                                     laserCloudIn->points[laserCloudIn->points.size() - 1].x) + 2 * M_PI;
        // 2π 

        // https://www.freesion.com/article/4533590429/
        // -π^ -- 0 -- ^π* -- 2π -- *3π     ^    * 
        //  -  > 3π   -π   3π   4π 
        //  -  <  π    π    π    π  
        //   2π     e-s  2π 
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
            segMsg.endOrientation -= 2 * M_PI;
        } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
        //  lego-loam    segMsg.orientationDiff  6.5 ~ 6.6   377°
    }

    // 3D point clou 2D range image
    void projectPointCloud(){
        // range image projection

        // verticalAngle  velodyne1  (-15°，+15° 
        // rowIdn (0,30 1  2 ) rin 

        float verticalAngle, horizonAngle, range, rangexy;
        size_t rowIdn, columnIdn, index, cloudSize; 
        PointType thisPoint;

        cloudSize = laserCloudIn->points.size();

        for (size_t i = 0; i < cloudSize; ++i){

            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;

            // find the row and column index in the iamge for this point
            //  rowIdn
            if (useCloudRing == true){
                rowIdn = laserCloudInRing->points[i].ring;
            }
            else{
                verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;  // 
            }
            // error
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            //  columnIdn
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            
            //roun 
            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;  // 
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;
            // error
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            //  range
            // range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y);
            rangexy = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y);      // only xy
            // range = rangexy;

            // error
            if (range < sensorMinimumRange)
                continue;
            
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            // rangeMatTemp.at<float>(rowIdn, columnIdn) = rangexy;

            //     2 。 intensit 。
            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            index = columnIdn  + rowIdn * Horizon_SCAN;         //  0-1800*16 
            fullCloud->points[index] = thisPoint;               // fullCloud  0-1800*16 ；
            
            // the corresponding range of a point is saved as "intensity"
            // fullInfoCloud fullCloud  fullInfoClou intensit range 
            fullInfoCloud->points[index] = thisPoint;
            fullInfoCloud->points[index].intensity = range;      
        }
    }

    // 16*180  8*180   
    //  groundMa 1；  labelma  -1.
    //    groundScanInd  
    void groundRemoval(){
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        // groundMat
        // -1, no valid info to check if ground of not
        //  0, initial value, after validation, means not ground
        //  1, ground
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            for (size_t i = 0; i < groundScanInd; ++i){

                lowerInd = j + ( i )*Horizon_SCAN;
                upperInd = j + (i+1)*Horizon_SCAN;
                
                // error
                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1){
                    // no info to check, invalid points
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }
                
                // handle
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                // 1   sensorMountAngle set 0
                if (abs(angle - sensorMountAngle) <= 10){
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }
        // extract ground cloud (groundMat == 1)
        // mark entry that doesn't need to label (ground and invalid point) for segmentation
        // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }
        if (pubGroundCloud.getNumSubscribers() != 0){
            for (size_t i = 0; i <= groundScanInd; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (groundMat.at<int8_t>(i,j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                }
            }
        }
    }


    void labelHandler(){
        labelMat_Hard = labelMat.clone();

        // 
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (labelMat.at<int>(i,j) == 0){
                    labelComponents(i, j);
                }
            }
        }

        // 
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (labelMat_Hard.at<int>(i,j) == 0){
                    labelComponentsForHard(i, j);
                }
            }
        }

    }


    // functio labelComponents(i, j  labelComponent ，cloudSegmentatio labelComponent    Ma segMs ”
    void cloudSegmentation(){





        // count  function start //
        int N_num_ground = 0;
        int N_num_usessd = 0;
        int N_num_abound = 0;

        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (labelMat.at<int>(i, j) == 999999){
                    N_num_abound += 1;
                }
                else if(labelMat.at<int>(i, j) == -1){
                    N_num_ground += 1;
                }
                else{
                    N_num_usessd +=1;
                }
            }
        }

        










        
        //  function ！！！
        labelMat_Copy = labelMat.clone();             // !  =  
        // cout << "labelMat_Copy start..." << endl;
        for(size_t i = 0; i < N_SCAN; ++i){
            for(size_t j = 0; j < Horizon_SCAN; ++j){
                if(labelMat_Copy.at<int>(i, j) == 999999){
                    labelMat_Copy.at<int>(i, j) = 0;

                    // cout << "check 01" << endl;
                    // cout << labelMat_Copy.at<int>(i, j) << endl;
                }
            }
            // cout << "labelMat_Copy over..." << endl;
        }
        

        //  labelMat  999999） 
        double Mat_max, Mat_min;
        cv::minMaxLoc(labelMat_Copy, &Mat_min, &Mat_max, 0, 0);
        int labelMat_NumMax = (int)Mat_max;
        // cout << labelMat_NumMax << " ; " << Mat_max << " ; " << Mat_min << endl;


        // 、Fla 
        for(int labelMat_Num = 1; labelMat_Num < (labelMat_NumMax + 1); ++labelMat_Num){

            bool labelMatFlag = false;

            // 
            for(size_t i = 0; i < N_SCAN; ++i){
                for(size_t j = 0; j < Horizon_SCAN; ++j){
                    if(labelMat.at<int>(i, j) == labelMat_Num){
                        // labelMatFlag = true;
                        
                        // only for test
                        // int labelMat_Num_Choose = labelMat_Num % 5;
                        // if(labelMat_Num_Choose == 0){
                        //     labelMatFlag = true;
                        // }

                        // int labelMat_Num_Choose = 3;
                        // if(labelMat_Hard.at<int>(i, j) == labelMat_Num_Choose){
                        //     labelMatFlag = true;
                            // cout << "check 01" << endl;
                        // }else{
                        //     // cout << "check 04" << endl;
                        // }


                        if((labelMat_Hard.at<int>(i, j) > 0) && (labelMat_Hard.at<int>(i, j) < 999999)){
                            labelMatFlag = true;
                            // cout <<  ！" << endl;
                        }

                    }
                }
            }

            // labelMat_Har ，labelMa  
            if(labelMatFlag == false){
                for(size_t i = 0; i < N_SCAN; ++i){
                    for(size_t j = 0; j < Horizon_SCAN; ++j){
                        if(labelMat.at<int>(i, j) == labelMat_Num){
                            labelMat.at<int>(i, j) = 999999;
                            // cout <<   ！" << endl;
                            // cout << "check 02" << endl;
                        }
                        // cout << "check 03" << endl;
                    }
                }
            }
        }





        //  






        
        // count  function start //
        int H_num_ground = 0;
        int H_num_usessd = 0;
        int H_num_abound = 0;

        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (labelMat_Hard.at<int>(i, j) == 999999){
                    H_num_abound += 1;
                }
                else if(labelMat_Hard.at<int>(i, j) == -1){
                    H_num_ground += 1;
                }
                else{
                    H_num_usessd +=1;
                }
            }
        }



        // count  function start //
        int C_num_ground = 0;
        int C_num_usessd = 0;
        int C_num_abound = 0;

        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (labelMat.at<int>(i, j) == 999999){
                    C_num_abound += 1;
                }
                else if(labelMat.at<int>(i, j) == -1){
                    C_num_ground += 1;
                }
                else{
                    C_num_usessd +=1;
                }
            }
        }

        // cout << "C_num_ground: " << C_num_ground << "\n" << "C_num_abound: " << C_num_abound << "\n" << "C_num_usessd: " << C_num_usessd << endl;
        // cout << "------------------" << endl;

        cout << "G: " << N_num_ground << "\n" << "N: " << N_num_usessd << "\n" << "H: " << H_num_usessd << "\n" << "C: " << C_num_usessd << endl;
        cout << "------------------" << endl;


        Count_full = Count_full + 1;
        N_num_ground_full = N_num_ground_full + N_num_ground;
        N_num_usessd_full = N_num_usessd_full + N_num_usessd;
        H_num_usessd_full = H_num_usessd_full + H_num_usessd;
        C_num_usessd_full = C_num_usessd_full + C_num_usessd;

        ros::Time beginning2(1317617735.038322);

        cout << "Time_Now: " << (ros::Time::now() - beginning2) << endl;
        cout << "Count_full: " << Count_full << endl;
        cout << "N_num_ground_full: " << N_num_ground_full << endl;
        cout << "N_num_usessd_full: " << N_num_usessd_full << endl;
        cout << "H_num_usessd_full: " << H_num_usessd_full << endl;
        cout << "C_num_usessd_full: " << C_num_usessd_full << endl;
        


        int sizeOfSegCloud = 0;
        // extract segmented cloud for lidar odometry
        for (size_t i = 0; i < N_SCAN; ++i) {

            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;  //4~19

            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                /  
                if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                    
                    / 
                    // outliers that will not be used for optimization (always continue)
                    if (labelMat.at<int>(i,j) == 999999){
                        if (i > groundScanInd && j % 5 == 0){
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                            continue;
                        }else{
                            continue;
                        }
                    }

                    / 
                    // majority of ground points are skipped
                    if (groundMat.at<int8_t>(i,j) == 1){
                        /  
                        if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                            continue;
                    }

                    // mark ground points so they will not be considered as edge features later  
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);

                    // mark the points' column index for marking occlusion later 
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;

                    // save range info
                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);

                    // save seg cloud
                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);

                    // size of seg cloud
                    ++sizeOfSegCloud;



                }
            }

            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }
        
        // extract segmented cloud for visualization SegmentedCloudPure segmentedCloudPur 
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            for (size_t i = 0; i < N_SCAN; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
                    }
                }
            }
        }
    }

    //   BFS
    //  3 ，labelCount++ 3 5     99999 。
    void labelComponents(int row, int col){
        // use std::queue std::vector std::deque will slow the program down greatly
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY;  

        bool lineCountFlag[N_SCAN] = {false};  / 

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;


        allPushedIndX[0] = row;     //  -> row
        allPushedIndY[0] = col;


        int allPushedIndSize = 1;

        
        //queueSize ，
        / whil 
        while(queueSize > 0){
            // Pop point
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];

            --queueSize;
            ++queueStartInd;

            // Mark popped point
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;


            // Loop through all the neighboring grids of popped grid
            / 
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
                // new index
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;

                // 
                // index should be within the boundary
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                
                //  
                // at range image margin (left or right side)
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;
                    
                // prevent infinite loop (caused by put already examined point back)
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                // , 
                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));

                / firs    
                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                else
                    alpha = segmentAlphaY;

                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));     //  
                // cout << "angle: " << angle << endl;
                //  
                if (angle > segmentTheta){

                    queueIndX[queueEndInd] = thisIndX;      //  -> thisIndX
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                    
                }

            }
        }


        //////////////////////////////////////////////////////////////////////////////
        ////                                                                 ////
        //////////////////////////////////////////////////////////////////////////////
        // 3  
        // check if this segment is valid  3  
        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)
            feasibleSegment = true;

        else if (allPushedIndSize >= segmentValidPointNum){
            // 3 5
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            if (lineCount >= segmentValidLineNum)
                //   
                feasibleSegment = true;            
        }

        // error
        // segment is valid, mark these points
        if (feasibleSegment == true){
            ++labelCount;
            // cout << "labelCount: " << labelCount << endl;
        }else{ // segment is invalid, mark these points
            // cout << "allPushedIndSize: " << allPushedIndSize << endl;
            for (size_t i = 0; i < allPushedIndSize; ++i){

                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }

    }


    void labelComponentsForHard(int row, int col){
        // use std::queue std::vector std::deque will slow the program down greatly
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY;  

        bool lineCountFlag[N_SCAN] = {false};  / 

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;


        allPushedIndX[0] = row;     //  -> row
        allPushedIndY[0] = col;


        int allPushedIndSize = 1;

        
        //queueSize ，
        / whil 
        while(queueSize > 0){
            // Pop point
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];

            --queueSize;
            ++queueStartInd;

            // Mark popped point
            labelMat_Hard.at<int>(fromIndX, fromIndY) = labelCount_Hard;

            // Loop through all the neighboring grids of popped grid
            / 
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
                // new index
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;

                // 
                // index should be within the boundary
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                
                //  
                // at range image margin (left or right side)
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;
                    
                // prevent infinite loop (caused by put already examined point back)
                if (labelMat_Hard.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                // , 
                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));

                / firs    
                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                else
                    alpha = segmentAlphaY;

                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));     //  
                // cout << "angle: " << angle << endl;

                if (angle > segmentTheta_Hard){

                    queueIndX[queueEndInd] = thisIndX;       
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat_Hard.at<int>(thisIndX, thisIndY) = labelCount_Hard;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                    
                }

            }
        }

        //////////////////////////////////////////////////////////////////////////////
        ////                                                                 ////
        //////////////////////////////////////////////////////////////////////////////
        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)
            feasibleSegment = true;

        else if (allPushedIndSize >= segmentValidPointNum_Hard){
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            if (lineCount >= segmentValidLineNum_Hard)
                feasibleSegment = true;            
        }

        // error
        if (feasibleSegment == true){
            ++labelCount_Hard;
        }else{
            for (size_t i = 0; i < allPushedIndSize; ++i){
                labelMat_Hard.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }


    
    void publishCloud(){
        // 1. Publish Seg Cloud Info
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);
        
        // 2. Publish clouds
        sensor_msgs::PointCloud2 laserCloudTemp;

        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);
        
        // segmented cloud with ground
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);
        
        // projected full cloud
        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }

        // original dense ground cloud
        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }

        // segmented cloud without ground
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }

        // projected full cloud info
        if (pubFullInfoCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }
    }
};




int main(int argc, char** argv){
    ros::init(argc, argv, "lego_loam");
    ImageProjection IP;
    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");
    ros::spin();
    return 0;
}
