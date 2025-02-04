#include "v_score.h"
V_SCORE::V_SCORE(Eigen::Vector3d camera_position, pcl::PointCloud<vPointType>::Ptr pointcloud) {
    initializeVariables(camera_position, pointcloud);

    denoisePointCloud();

    // estimateNormals();

    setVectorSize(); 

    calculateScore();


}

void V_SCORE::initializeVariables(Eigen::Vector3d camera_position, pcl::PointCloud<vPointType>::Ptr pointcloud) {
    // 올바른 포인터 생성
    original_cloud_ = pcl::PointCloud<vPointType>::Ptr(new pcl::PointCloud<vPointType>());
    denoise_cloud_ = pcl::PointCloud<vPointType>::Ptr(new pcl::PointCloud<vPointType>());
    projected_cloud_ = pcl::PointCloud<vPointType>::Ptr(new pcl::PointCloud<vPointType>());
    // normals_ = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());

    // 포인터 할당 (기존 포인트 클라우드와 동일한 데이터 참조)
    camera_position_ = camera_position;
    original_cloud_ = pointcloud;  // ✅ 포인터 복사 (공유)
}

void V_SCORE::denoisePointCloud(){
    pcl::StatisticalOutlierRemoval<vPointType> sor;
    sor.setInputCloud(original_cloud_);
    sor.setMeanK(3);
    sor.setStddevMulThresh(0.8);
    sor.filter(*denoise_cloud_);
    // std::cout << "Denoising completed, noise points: " <<denoise_cloud_->size() << std::endl;
}

// void V_SCORE::estimateNormals(){
//     pcl::NormalEstimation<vPointType, pcl::Normal> ne;
//     ne.setInputCloud(denoise_cloud_);
//     pcl::search::KdTree<vPointType>::Ptr tree(new pcl::search::KdTree<vPointType>());
//     ne.setSearchMethod(tree);
//     ne.setKSearch(10);  // Use 10 neighbors
//     ne.compute(*normals_);
// }

void V_SCORE::setVectorSize(){
    int denoise_size = denoise_cloud_->points.size();
    original_dist_vec_.resize(denoise_size);
    vis_score_vec_.resize(denoise_size);
    normal_score_vec_.resize(denoise_size);
    planar_score_vec_.resize(denoise_size);
    mix_score_vec_.resize(denoise_size);
    color_vec_.resize(denoise_size);
}

void V_SCORE::projectPointCloud(){
    *projected_cloud_ = *denoise_cloud_;
    for (int idx=0; idx<projected_cloud_->points.size(); idx++){
        auto &point = projected_cloud_->points[idx];
        double x = point.x - camera_position_[0];
        double y = point.y - camera_position_[1];
        double z = point.z - camera_position_[2];

        double distance = sqrt(x*x + y*y);
        original_dist_vec_[idx] = distance;
        double ratio = 5.0/distance;

        // project all point to screen space whose distance is 1.0 meter
        point.x = x * ratio + camera_position_[0];
        point.y = y * ratio + camera_position_[1];
        point.z = z * ratio + camera_position_[2];
    }
    // set kdtree 
    projected_kdtree_->setInputCloud(projected_cloud_);

    // projected_kdtree_.SetGeometry(*projected_cloud_);
    
}

void V_SCORE::projectPointCloud(pcl::PointCloud<vPointType>::Ptr in_cloud, Eigen::Vector3d camera_position, pcl::search::KdTree<vPointType>::Ptr in_tree, std::vector<double> &dist_vec){
    pcl::PointCloud<vPointType>::Ptr projected_cloud;
    projected_cloud = in_cloud;
    dist_vec.resize(projected_cloud->points.size());
    for (int idx=0; idx<projected_cloud->points.size(); idx++){
        auto &point = projected_cloud->points[idx];
        double x = point.x - camera_position[0];
        double y = point.y - camera_position[1];
        double z = point.z - camera_position[2];

        double distance = sqrt(x*x + y*y);
        dist_vec[idx] = distance;
        double ratio = 5.0/distance;

        // project all point to screen space whose distance is 1.0 meter
        // point.x = x * ratio + camera_position_[0];
        // point.y = y * ratio + camera_position_[1];
        // point.z = z * ratio + camera_position_[2];
        point.x = x * ratio;
        point.y = y * ratio;
        point.z = z * ratio;
    }
    // set kdtree 
    // in_tree.SetGeometry(projected_cloud);
    in_tree->setInputCloud(projected_cloud);

}

void V_SCORE::projectPointCloud(pcl::PointCloud<vPointType>::Ptr &in_cloud, Eigen::Vector3d camera_position){
    for (int idx=0; idx<in_cloud->points.size(); idx++){
        auto &point = in_cloud->points[idx];
        double x = point.x - camera_position[0];
        double y = point.y - camera_position[1];
        double z = point.z - camera_position[2];

        double distance = sqrt(x*x + y*y);
        double ratio = 5.0/distance;

        // project all point to screen space whose distance is 1.0 meter
        point.x = x * ratio + camera_position_[0];
        point.y = y * ratio + camera_position_[1];
        point.z = z * ratio + camera_position_[2];
    }
}


void V_SCORE::calculateScore(){
    for (int idx=0; idx<denoise_cloud_->points.size(); idx++){
        double normal_score = calculateNormalScore(idx);
        double mix_score = normal_score;
        mix_score_vec_[idx] = mix_score;
    }
}

inline double V_SCORE::calculateNormalScore(int idx){
    Eigen::Vector3d point_eigen(
        denoise_cloud_->points[idx].x,
        denoise_cloud_->points[idx].y,
        denoise_cloud_->points[idx].z
    );
    Eigen::Vector3d los = point_eigen - camera_position_;
    double normal_score = 0;
    auto nor_pcl = denoise_cloud_->points[idx];
    Eigen::Vector3d normal = {nor_pcl.normal_x, nor_pcl.normal_y, nor_pcl.normal_z};

    Eigen::Matrix3d z_filter_mat;
    // z_filter_mat << 1,0,0,0,1,0,0,0,0;
    z_filter_mat << 1,0,0,0,1,0,0,0,1;
    // los = z_filter_mat * los;
    double a = los.transpose() * ( z_filter_mat * normal );
    double angle_nl = acos(abs(a / (los.norm() * normal.norm())));

    // below code is not needed, since the result of acos is in range of [0, pi] 
    
    // Also, since input of acos is always in range of [0,1], angle_nl is in range of [0,pi/2]

    // if (angle_nl > 2*M_PI)
    //     angle_nl -= 2*M_PI;
    // else if (angle_nl < 0)
    //     angle_nl += 2*M_PI;

    //since angle_nl is in range of [0,pi/2], normal_score is in range of [0,1]
    normal_score = cos(angle_nl);

    normal_score_vec_[idx] = normal_score;
    return normal_score;
}

inline double V_SCORE::calculateVisibilityScore(int idx){
    // std::vector<int> indices_vec;
    // std::vector<double> dists_vec;
    // auto point = projected_cloud_->points[idx];
    // projected_kdtree_.SearchRadius(point,1,indices_vec,dists_vec);
    // // projected_kdtree_.SearchHybrid(point,0.1,4,indices_vec,dists_vec);

    // double min_dist = std::numeric_limits<double>::max();
    // double max_dist = std::numeric_limits<double>::min();

    // for (int n_idx=1; n_idx<indices_vec.size(); n_idx++){
    //     double cur_dist = original_dist_vec_[indices_vec[n_idx]];
        
    //     if (cur_dist > max_dist){
    //         max_dist = cur_dist;
    //     }
    //     if (cur_dist < min_dist){
    //         min_dist = cur_dist;
    //     }
    // }

    // double a1 = original_dist_vec_[idx] - min_dist;
    // double a2 = max_dist-min_dist;
    // double v_score = exp(-1 * pow(a1,2) / pow(a2,2));

    // // double a1 = pow(original_dist_vec_[idx] - min_dist,2);
    // // double a2 = pow(min_dist,2);
    // // double v_score = exp((original_dist_vec_[idx] - min_dist)*(-0.6)/(0.05*min_dist));

    // if(debug_file_)
    //     vis_score_file_ << v_score << std::endl;

    // vis_score_vec_[idx] = v_score;

    return 0.0 ;
    // return v_score;
}

inline double V_SCORE::calculatePlanarScore(int idx){
    // assert(denoise_cloud_->HasCovariances());

    // eigen_solver_.compute(denoise_cloud_->covariances_[idx], Eigen::ComputeEigenvectors);
    // auto eigen_values = eigen_solver_.eigenvalues();
    // double edge_score = abs(sqrt(abs(eigen_values[2]))-sqrt(abs(eigen_values[1])))/sqrt(abs(eigen_values[2]));
    // double planar_score = abs(sqrt(abs(eigen_values[1]))-sqrt(abs(eigen_values[0])))/sqrt(abs(eigen_values[2]));

    // planar_score_vec_[idx] = edge_score + planar_score;
    // return edge_score + planar_score;
    return 0.0;
}

void V_SCORE::colorizeResult(){
    // double max_score = *max_element(mix_score_vec_.begin(), mix_score_vec_.end());
    // double min_score = *min_element(mix_score_vec_.begin(), mix_score_vec_.end());
    double max_score = 1.0;
    // double min_score = 0.0;
    // double diff = max_score - min_score;

    for (int idx=0; idx<mix_score_vec_.size(); idx++){
        auto score = mix_score_vec_[idx];
        // double ratio = 2*(score-min_score)/diff;
        double ratio = 2*score;
        // Closer to Red, Lower value.
        double r = ((1-ratio) < 0) ? 0.0 : (1-ratio);
        double b = ((ratio - 1) < 0) ? 0.0 : (ratio-1);
        double g = 1-b-r;

        color_vec_[idx] = {r,g,b};
    }

    // denoise_cloud_->colors_ = color_vec_;
}

void V_SCORE::visualizeResult(){
    // auto pose_sphere = geometry::TriangleMesh::CreateSphere(2);
    // pose_sphere->PaintUniformColor({0,1,0});
    // pose_sphere->Translate(camera_position_);
    // visualization::DrawGeometries({denoise_cloud_},"V_SCORE RESULT",1920,1080,50,50);
}


void V_SCORE::setFile(std::string path){
    vis_score_file_.open("/home/hmc/V_ICP/file/vis.txt");		
    normal_score_file_.open("/home/hmc/V_ICP/file/normal.txt");	
    mix_score_file_.open("/home/hmc/V_ICP/file/mix.txt");
}

void V_SCORE::closeFile(){
    vis_score_file_.close();
    normal_score_file_.close();
    mix_score_file_.close();
}

inline double V_SCORE::getScore(int idx){
    return mix_score_vec_[idx];
}

std::vector<double> V_SCORE::getScores(){
    return mix_score_vec_;
}

pcl::PointCloud<vPointType>::Ptr V_SCORE::getCloud(){
    return denoise_cloud_;
}

typedef std::pair<double,int> val_idx;
bool comparator ( const val_idx& l, const val_idx& r) {      
return l.first > r.first;
}

std::vector<int> V_SCORE::getTopIndex(int num){
    // assert(num < mix_score_vec_.size());

    // std::vector<val_idx> tmp;
    // // #pragma omp parallel for
    // for(int i=0; i<mix_score_vec_.size(); i++){
    // // #pragma omp critical
    //     tmp.push_back(val_idx(mix_score_vec_[i],i));
    // }

    // std::stable_sort(tmp.begin(),tmp.end(),comparator);
    std::vector<int> sorted_idx;
    // for(int i=0; i<num; i++){
    //     sorted_idx.push_back(tmp[i].second);
    // }
    // denoise_cloud_->colors_ = color_vec_;
    // std::vector<Eigen::Vector3d> test;
    // for(int ii:sorted_idx){
    //     test.push_back(denoise_cloud_->points[ii]);
    // }
    // pcl::PointCloud<vPointType>::Ptr test_cloud_(new pcl::PointCloud<vPointType>());;
    // *test_cloud_ = test;
    // visualization::DrawGeometries({test_cloud_});
    return sorted_idx;
}

pcl::PointCloud<vPointType>::Ptr V_SCORE::getTopScoredCloud(int num, std::vector<double> &scores, pcl::PointCloud<pcl::Normal>::Ptr normals){
    assert(num < mix_score_vec_.size());
    std::vector<val_idx> tmp;
    // #pragma omp parallel for
    for(int i=0; i<mix_score_vec_.size(); i++){
    // #pragma omp critical
        tmp.push_back(val_idx(mix_score_vec_[i],i));
    }

    pcl::PointCloud<vPointType>::Ptr test_cloud_(new pcl::PointCloud<vPointType>());

    std::stable_sort(tmp.begin(),tmp.end(),comparator);
    for(int i=0; i<num; i++){
        test_cloud_->points.push_back(denoise_cloud_->points[tmp[i].second]);  // ✅ 바로 추가
        scores.push_back(mix_score_vec_[tmp[i].second]);
        normals->points.push_back(normals_->points[tmp[i].second]);
    }

    return test_cloud_;
}

pcl::PointCloud<vPointType>::Ptr V_SCORE::getTopScoredCloud(int num, std::vector<double> &scores){
    assert(num < mix_score_vec_.size());
    std::vector<val_idx> tmp;
    // #pragma omp parallel for
    for(int i=0; i<mix_score_vec_.size(); i++){
    // #pragma omp critical
        tmp.push_back(val_idx(mix_score_vec_[i],i));
    }

    pcl::PointCloud<vPointType>::Ptr test_cloud_(new pcl::PointCloud<vPointType>());

    std::stable_sort(tmp.begin(),tmp.end(),comparator);
    for(int i=0; i<num; i++){
        test_cloud_->points.push_back(denoise_cloud_->points[tmp[i].second]);  // ✅ 바로 추가
        scores.push_back(mix_score_vec_[tmp[i].second]);
    }
    return test_cloud_;
}

pcl::PointCloud<vPointType>::Ptr V_SCORE::getTopScoredCloud(double ratio, std::vector<double> &scores){
    assert(ratio <= 1.0);
    int num = static_cast<int>(denoise_cloud_->points.size() * ratio) ;

    std::vector<val_idx> tmp;
    // #pragma omp parallel for
    for(int i=0; i<mix_score_vec_.size(); i++){
    // #pragma omp critical
        tmp.push_back(val_idx(mix_score_vec_[i],i));
    }

    pcl::PointCloud<vPointType>::Ptr test_cloud_(new pcl::PointCloud<vPointType>());

    std::stable_sort(tmp.begin(),tmp.end(),comparator);
    for(int i=0; i<num; i++){
        test_cloud_->points.push_back(denoise_cloud_->points[tmp[i].second]);  // ✅ 바로 추가
        scores.push_back(mix_score_vec_[tmp[i].second]);
    }
    return test_cloud_;
}


pcl::PointCloud<vPointType>::Ptr V_SCORE::getTopScoredCloud(int num, pcl::PointCloud<vPointType>::Ptr pointcloud, std::vector<double> weight){
    assert(pointcloud->points.size() == weight.size());

    std::vector<val_idx> tmp;
    // #pragma omp parallel for
    for(int i=0; i<weight.size(); i++){
    // #pragma omp critical
        tmp.push_back(val_idx(weight[i],i));
    }

    std::stable_sort(tmp.begin(),tmp.end(),comparator);
    std::vector<int> sorted_idx;
    for(int i=0; i<num; i++){
        sorted_idx.push_back(tmp[i].second);
    }
    
    std::vector<Eigen::Vector3d> test;
    for(int ii:sorted_idx){
        // test.push_back(pointcloud->points[ii]);
    }
    pcl::PointCloud<vPointType>::Ptr test_cloud_(new pcl::PointCloud<vPointType>());;
    // *test_cloud_ = test;
    // visualization::DrawGeometries({test_cloud_});
    return test_cloud_;
}

pcl::PointCloud<vPointType>::Ptr V_SCORE::getTopScoredCloudByScore(double score){

    // denoise_cloud_->colors_ = color_vec_;

    // std::vector<Eigen::Vector3d> top_scored_points;

    // for(int i=0; i<mix_score_vec_.size(); i++){

    //     if(mix_score_vec_[i] > score){

    //     top_scored_points.push_back(denoise_cloud_->points[i]);

    //     }
    // }
    pcl::PointCloud<vPointType>::Ptr test_cloud_(new pcl::PointCloud<vPointType>());;
    // *test_cloud_ = top_scored_points;
    return test_cloud_;
}

std::vector<double> V_SCORE::getCrossScores( std::vector<Eigen::Vector3d> &src_pt, std::vector<Eigen::Vector3d> &tgt_pt, Eigen::Vector3d src_pose, Eigen::Vector3d tgt_pose){
    using namespace Eigen;
    assert(src_pt.size() == tgt_pt.size());

    int N = src_pt.size();
    Map<Matrix3Xd> src(&src_pt[0].x(),3,N); //maps vector<Vector3d>
    Map<Matrix3Xd> tgt(&tgt_pt[0].x(),3,N);

    Matrix3Xd los_src_pt_to_tgt;
    Matrix3Xd los_tgt_pt_to_src;
    
    los_src_pt_to_tgt = src.colwise() - tgt_pose; 
    los_tgt_pt_to_src = tgt.colwise() - src_pose;

    Eigen::Matrix3d z_filter_mat;
    z_filter_mat << 1,0,0,0,1,0,0,0,0;
    MatrixXd a = los_tgt_pt_to_src.transpose() * ( z_filter_mat * los_src_pt_to_tgt );
    std::vector<double> cross_score_vec;
    // #pragma omp for
    for (int i=0;i<N;i++){
        auto cross_score = acos(a(i,i) / (los_src_pt_to_tgt.col(i).norm() * los_tgt_pt_to_src.col(i).norm()));
        
        if (cross_score > 2*M_PI)
            cross_score -= 2*M_PI;
        else if (cross_score < 0)
            cross_score += 2*M_PI;
        cross_score = 1 - (cross_score/M_PI);
        cross_score_vec.push_back(cross_score);
    }

    return cross_score_vec;
}

void V_SCORE::visualizeWeight(const pcl::PointCloud<vPointType>::Ptr pointcloud, const std::vector<double> weight){
    // // assert(pointcloud.size() ==  weight.size());

    // pcl::PointCloud<vPointType>::Ptr tmp_cloud(new pcl::PointCloud<vPointType>());;
    // *tmp_cloud = *pointcloud;

    // std::vector<Eigen::Vector3d> tmp_color_vec(pointcloud->points.size());
    // double max_score = 1.0;
    // double min_score = 0.0;
    // double diff = max_score - min_score;

    // for (int idx=0; idx<weight.size(); idx++){
    //     auto score = weight[idx];
    //     double ratio = 2*(score-min_score)/diff;
    //     // Closer to Red, Lower value.
    //     double r = ((1-ratio) < 0) ? 0.0 : (1-ratio);
    //     double b = ((ratio - 1) < 0) ? 0.0 : (ratio-1);
    //     double g = 1-b-r;

    //     tmp_color_vec[idx] = {r,g,b};
    // }

    // denoise_cloud_->PaintUniformColor({0,0,0});
    // tmp_cloud->colors_ = tmp_color_vec;
    // visualization::DrawGeometries({denoise_cloud_,tmp_cloud},"Visualize Weight",1920,1080,50,50);
    // visualization::DrawGeometries({tmp_cloud},"Visualize Weight",1920,1080,50,50);
}

void V_SCORE::visualizeWeight(const pcl::PointCloud<vPointType>::Ptr pointcloud, const std::vector<double> weight,const pcl::PointCloud<vPointType>::Ptr pointcloud2){
    // // assert(pointcloud.size() ==  weight.size());

    // pcl::PointCloud<vPointType>::Ptr tmp_cloud(new pcl::PointCloud<vPointType>());;
    // *tmp_cloud = *pointcloud;

    // std::vector<Eigen::Vector3d> tmp_color_vec(pointcloud->points.size());
    // double max_score = 1.0;
    // double min_score = 0.0;
    // double diff = max_score - min_score;

    // for (int idx=0; idx<weight.size(); idx++){
    //     auto score = weight[idx];
    //     double ratio = 2*(score-min_score)/diff;
    //     // Closer to Red, Lower value.
    //     double r = ((1-ratio) < 0) ? 0.0 : (1-ratio);
    //     double b = ((ratio - 1) < 0) ? 0.0 : (ratio-1);
    //     double g = 1-b-r;

    //     tmp_color_vec[idx] = {r,g,b};
    // }

    // denoise_cloud_->PaintUniformColor({0,0,0});
    // tmp_cloud->colors_ = tmp_color_vec;
    // pointcloud2->PaintUniformColor({0,0,0});
    // visualization::DrawGeometries({tmp_cloud,pointcloud2},"Visualize Weight",1920,1080,50,50);
}

std::vector<double> V_SCORE::getUncertainVisiblityScores(const pcl::PointCloud<vPointType>::Ptr source_cloud, const pcl::PointCloud<vPointType>::Ptr target_cloud, Eigen::Vector3d position, const double std_score, const double std_perc){
    pcl::PointCloud<vPointType>::Ptr fused_cloud(new pcl::PointCloud<vPointType>);
    pcl::search::KdTree<vPointType>::Ptr fused_kdtree(new pcl::search::KdTree<vPointType>());

    *fused_cloud = *source_cloud;
    *fused_cloud += *target_cloud;

    std::vector<double> visibility_scores_vec(source_cloud->points.size());
    std::vector<double> point_dist_vec;
    // // std::cout << "position" << std::endl;
    // // for (int i=0; i<3 ; i++)
    // //     std::cout << position[i] << " " << std::endl;
    projectPointCloud(fused_cloud,position,fused_kdtree, point_dist_vec);
    pcl::PointCloud<vPointType>::Ptr projected_source_cloud(new pcl::PointCloud<vPointType>);
    *projected_source_cloud = *source_cloud;
    // pcl::PointCloud<vPointType>::Ptr projected_source_cloud  = source_cloud;
    projectPointCloud(projected_source_cloud, position);

    std::vector<int> indices_vec;
    std::vector<float> dists_vec;
    // // std::cout << point_dist_vec.size() << std::endl;
    // // std::cout << projected_source_cloud->points.size() << std::endl;
    for (int idx=0; idx<projected_source_cloud->points.size(); idx++){
        auto point = projected_source_cloud->points[idx];
    //     // fused_kdtree.SearchHybrid(point,3.0,6,indices_vec,dists_vec);
        // fused_kdtree.nearestKSearch(point,0.1,indices_vec,dists_vec);
        double min_dist = std::numeric_limits<double>::max();
        double max_dist = std::numeric_limits<double>::min();
        if(fused_kdtree->radiusSearch(point, 0.1, indices_vec, dists_vec)>0){

            for (int n_idx=1; n_idx<indices_vec.size(); n_idx++){
                double cur_dist = point_dist_vec[indices_vec[n_idx]];
            
                if (cur_dist >= max_dist){
                    max_dist = cur_dist;
                }
                if (cur_dist <= min_dist){
                    min_dist = cur_dist;
                }
            }
        }
        // double a1 = pow(point_dist_vec[idx] - min_dist,2);

        double a1 = point_dist_vec[idx] - min_dist;
        // double a2 = pow(max_dist-min_dist,2);
        double a2 = min_dist;
        // double scala = pow(1-(min_dist/max_dist),1.5) * (max_dist-min_dist);
        // double scala = pow(200.0/min_dist,1.5);
        // double v_score = exp(-scala * a1 / a2);
        // double v_score = exp(-(point_dist_vec[idx]- min_dist)*12/min_dist);  
        // std::cout << "score : " << std_score << " perc : " << std_perc;

        double v_score = exp((a1 * -0.6) / (a2 * 0.05));
        // double v_score = exp((a1 * log(std_score)) / (a2 * std_perc));
        visibility_scores_vec[idx] = std::min(v_score,1.0);

        // visibility_scores_vec[idx] = v_score;
    }

    return visibility_scores_vec;
}