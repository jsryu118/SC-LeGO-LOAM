#include <iostream>       
#include <typeinfo>      
#include <ctime>
#include <fstream>
#include <cmath>
#include <tgmath.h>
#include <algorithm>
#include <numeric>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>  // ✅ PCD io header
#include <pcl/registration/transformation_estimation_point_to_plane_lls_weighted.h>
#include <pcl/registration/correspondence_estimation.h>
// #include <pcl_conversions/pcl_conversions.h>
#include "v_score.h"
// #include "icp_closedform.h"
// #include "progressbar.h"

class ViewpointAwareICP{
private:
    using vPointType = pcl::PointXYZINormal;

    pcl::PointCloud<vPointType>::Ptr targetCloud_;
    pcl::PointCloud<vPointType>::Ptr sourceCloud_;
    

    Eigen::Isometry3d final_transformation_ = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d initialGuess_;
    Eigen::Isometry3d inv_initialGuess_;
    

    bool initSrc_ = false;
    bool initTgt_ = false;
    bool initGuess_ = false;

    bool converged_ = false;

    double corres_dist_ = 10.0;
    int max_iter_ = 200;
    double eps_transform_ = 1e-6;
    double eps_euclidean_fitness_ = 1e-6;
    // void estimateNormalVec();
    bool second_stage = false;
    double fitness_score_ = std::numeric_limits<double>::max();


    
public:
    ViewpointAwareICP();

    void setInputTarget(pcl::PointCloud<pcl::PointXYZI>::Ptr);
    void setInputSource(pcl::PointCloud<pcl::PointXYZI>::Ptr);
    void setInitialGuess(Eigen::Isometry3d);
    // pcl::PointCloud<vPointType>::Ptr computeNormals(pcl::PointCloud<vPointType>::Ptr cloud);
    pcl::PointCloud<vPointType>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);

    double getFitnessScoreNupdateWeights(pcl::CorrespondencesPtr corres, std::vector<double> weights);
    double getFitnessScoreNupdateUniformWeights(pcl::CorrespondencesPtr corres);


    void setMaxCorrespondenceDistance(double corres_dist) {corres_dist_ = corres_dist;}
    void setMaximumIterations(int max_iter) {max_iter_ = max_iter;}
    void setTransformationEpsilon(double eps_transform) {eps_transform_ = eps_transform;}
    void setEuclideanFitnessEpsilon(double eps_euclidean_fitness) {eps_euclidean_fitness_ = eps_euclidean_fitness;}
    void align(pcl::PointCloud<vPointType>& result_pc);
    void transformPointCloud(pcl::PointCloud<vPointType>::Ptr cloudIn, Eigen::Matrix4d transformIn);

    bool hasConverged() {return converged_;}
    double getFitnessScore() {return fitness_score_;}
    Eigen::Isometry3d getFinalTransformation() {return final_transformation_;}
};