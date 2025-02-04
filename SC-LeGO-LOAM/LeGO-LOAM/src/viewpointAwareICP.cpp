#include "viewpointAwareICP.h"

ViewpointAwareICP::ViewpointAwareICP(){
    targetCloud_ = pcl::PointCloud<vPointType>::Ptr(new pcl::PointCloud<vPointType>());
    sourceCloud_ = pcl::PointCloud<vPointType>::Ptr(new pcl::PointCloud<vPointType>());
}

pcl::PointCloud<vPointType>::Ptr ViewpointAwareICP::computeNormals(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
    ne.setSearchMethod(tree);
    ne.setKSearch(10);
    ne.compute(*normals);

    pcl::PointCloud<vPointType>::Ptr cloud_with_normals(new pcl::PointCloud<vPointType>());
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
    
    return cloud_with_normals;
}

void ViewpointAwareICP::setInputTarget(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    if (!cloud || cloud->empty()) {
        std::cerr << "Input target cloud is empty!" << std::endl;
        return;
    }
    targetCloud_ = computeNormals(cloud);
    initTgt_ = true;
}

void ViewpointAwareICP::setInputSource(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    if (!cloud || cloud->empty()) {
        std::cerr << "Input source cloud is empty!" << std::endl;
        return;
    }
    sourceCloud_ = computeNormals(cloud);
    initSrc_ = true;
}


void ViewpointAwareICP::setInitialGuess(Eigen::Isometry3d initialGuess) {
    initialGuess_ = initialGuess;
    inv_initialGuess_ = initialGuess_.inverse();
    initGuess_ = true;
}

void ViewpointAwareICP::transformPointCloud(pcl::PointCloud<vPointType>::Ptr cloudIn, Eigen::Matrix4d transform_matrix){
    for (auto& pt : cloudIn->points) {
        Eigen::Vector4d point(pt.x, pt.y, pt.z, 1.0);  // Homogeneous 좌표로 변환
        Eigen::Vector4d transformed_point = transform_matrix * point;  // 변환 적용
        
        pt.x = transformed_point.x();
        pt.y = transformed_point.y();
        pt.z = transformed_point.z();
    }
}

double ViewpointAwareICP::getFitnessScoreNupdateWeights(pcl::CorrespondencesPtr corres, std::vector<double> weights){
    double sum_distance = 0.0;

    for (size_t i = 0; i < corres->size(); ++i)
    {
        float d = corres->at(i).distance;
        corres->at(i).weight = static_cast<float>(weights[corres->at(i).index_match]);;
        sum_distance += std::sqrt(d);
    }

    return sum_distance / static_cast<double>(corres->size());
}

double ViewpointAwareICP::getFitnessScoreNupdateUniformWeights(pcl::CorrespondencesPtr corres){
    double sum_distance = 0.0;

    for (size_t i = 0; i < corres->size(); ++i)
    {
        float d = corres->at(i).distance;
        corres->at(i).weight = 1.0f;
        sum_distance += std::sqrt(d);
    }

    return sum_distance / static_cast<double>(corres->size());
}


void ViewpointAwareICP::align(pcl::PointCloud<vPointType>& result_pc) {
    if (!initSrc_ || !initTgt_ || !initGuess_) {
        std::cerr << "Source or Target not initialized!" << std::endl;
        return;
    }

    pcl::PointCloud<vPointType>::Ptr temp(new pcl::PointCloud<vPointType>());
    pcl::transformPointCloudWithNormals(*targetCloud_, *temp, inv_initialGuess_.matrix());
    targetCloud_ = temp;

    V_SCORE score_s(Eigen::Vector3d::Zero(), sourceCloud_);
    V_SCORE score_t(Eigen::Vector3d::Zero(), targetCloud_);

    std::vector<double> src_scores, tgt_scores;

    *sourceCloud_ = *score_s.getTopScoredCloud(0.7, src_scores);
    *targetCloud_ = *score_t.getTopScoredCloud(0.7, tgt_scores); 

    pcl::search::KdTree<vPointType>::Ptr target_kdtree_(new pcl::search::KdTree<vPointType>());
    target_kdtree_->setInputCloud(targetCloud_);

    Eigen::Isometry3d last_result = Eigen::Isometry3d::Identity();
    std::vector<Eigen::Vector3d> src_points, tgt_points, normals;
    std::vector<std::pair<int,int>> corres_pairs;
    std::vector<double> valid_scores;

	pcl::registration::CorrespondenceEstimation<vPointType, vPointType> corr_est;
	corr_est.setInputTarget(targetCloud_);
    pcl::PointCloud<vPointType>::Ptr current_source(new pcl::PointCloud<vPointType>());
    pcl::copyPointCloud(*sourceCloud_, *current_source);

    // Weighted ICP Iteration
	Eigen::Matrix4f T_last = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f T_total = Eigen::Matrix4f::Identity();
    pcl::registration::TransformationEstimationPointToPlaneLLSWeighted<vPointType, vPointType>::Ptr transformation_estimation(
        new pcl::registration::TransformationEstimationPointToPlaneLLSWeighted<vPointType, vPointType>());

    int iter = 0;
    while (iter < max_iter_){
        corr_est.setInputSource(current_source);
        pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
        corr_est.determineCorrespondences(*correspondences, corres_dist_);

        if (correspondences->empty())
        {
            std::cout << "No correspondences found at iteration " << iter << std::endl;
            break;
        }
        // std::vector<float> weights(targetCloud_->points.size(), 1.0f);

        fitness_score_ = getFitnessScoreNupdateUniformWeights(correspondences);
        // fitness_score_ = getFitnessScoreNupdateWeights(correspondences, tgt_scores);

        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        transformation_estimation->estimateRigidTransformation(*current_source, *targetCloud_, *correspondences, delta);

        // 4. 누적 변환 업데이트: T_total = delta * T_total;
        T_total = delta * T_total;

        // 5. 현재 소스 클라우드에 delta 변환 적용하여 업데이트
        pcl::PointCloud<vPointType>::Ptr temp(new pcl::PointCloud<vPointType>());
        pcl::transformPointCloudWithNormals(*current_source, *temp, delta);
        current_source = temp;



        // 6. 변화량(Transformation Difference)을 측정 (예: delta에서 Identity의 차이)
        Eigen::Matrix4f diff = delta - T_last;
		T_last = delta;
        double trans_diff_norm = diff.norm();

        std::cout << "Iteration " << iter+1 
                  << ", transformation diff norm: " << trans_diff_norm 
                  << ", fitness score: " << fitness_score_ << std::endl;

        // 7. 종료 조건 검사: 변환 변화량과 피트니스가 모두 충분히 작으면 종료
        if (second_stage && trans_diff_norm < 0.00001)
        // if (trans_diff_norm < eps_transform_ && fitness_score_ < eps_euclidean_fitness_)
            break;
        
        // if (trans_diff_norm < eps_transform_*2 && fitness_score_ < eps_euclidean_fitness_*2){
        if (trans_diff_norm < 0.0001){
            // break;     
            // auto uv_score = score_s.getUncertainVisiblityScores(sourceCloud_, targetCloud_);
            // auto tgt_uv_score = score_s.getUncertainVisiblityScores(targetCloud_,sourceCloud_);

            second_stage = true;
            std::cout << "second" << std::endl;


        }
        if(second_stage){
            auto tgt_uv_score = score_t.getUncertainVisiblityScores(targetCloud_,sourceCloud_,T_total.cast<double>().block<3,1>(0,3));
            tgt_scores = tgt_uv_score;
        }

        iter++;
    }

    transformPointCloud(sourceCloud_, initialGuess_.matrix());
    final_transformation_.matrix() = final_transformation_.cast<double>() * initialGuess_.matrix();
    // final_transformation_ = final_transformation_ * initialGuess_;
    result_pc = *sourceCloud_; 
}
