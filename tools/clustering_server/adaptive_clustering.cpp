#include <cmath>
#include <iostream>
#include "adaptive_clustering.h"

AdaptiveClustering::AdaptiveClustering(std::string sensor_model) : sensor_model_(sensor_model) {
  if(sensor_model_.compare("VLP-16") == 0) {
    regions_[0] = 2; regions_[1] = 3; regions_[2] = 3; regions_[3] = 3; regions_[4] = 3;
    regions_[5] = 3; regions_[6] = 3; regions_[7] = 2; regions_[8] = 3; regions_[9] = 3;
    regions_[10]= 3; regions_[11]= 3; regions_[12]= 3; regions_[13]= 3;
  } else if (sensor_model_.compare("HDL-32E") == 0) {
    regions_[0] = 4; regions_[1] = 5; regions_[2] = 4; regions_[3] = 5; regions_[4] = 4;
    regions_[5] = 5; regions_[6] = 5; regions_[7] = 4; regions_[8] = 5; regions_[9] = 4;
    regions_[10]= 5; regions_[11]= 5; regions_[12]= 4; regions_[13]= 5;
  } else if (sensor_model_.compare("HDL-64E") == 0) {
    regions_[0] = 14; regions_[1] = 14; regions_[2] = 14; regions_[3] = 15; regions_[4] = 14;
  } else {
    std::cout << "Sensor model does not exist!" << std::endl;
  }


  float tolerance = 0;
  for(int r = 0; r < AdaptiveClustering::region_max_; r++) {
    tolerance += 0.1;
    extractClusterParam_t ecp;
    ecp.voxelX = tolerance;
    ecp.voxelY = tolerance;
    ecp.voxelZ = tolerance;

    ecp.minClusterSize = cluster_size_min;
    ecp.maxClusterSize = cluster_size_max;
    ecp.countThreshold = 0;

    params.push_back(ecp);
  }

}

AdaptiveClustering::~AdaptiveClustering(){
  //cudaStreamDestroy(stream);
}

std::vector<std::vector<float>> AdaptiveClustering::cluster(
    const float* inp_pc, const unsigned inp_pc_size){ // input point cloud, x y z i format
  // Divide the point cloud into nested circular regions centred at the sensor.
  // For more details, see our IROS-17 paper 
  // "Online learning for human classification in 3D LiDAR-based tracking"

  // Filtering here takes only 1 ms
  std::vector<float> filtered_pc;
  filtered_pc.reserve(inp_pc_size);

  for(auto i=0; i<inp_pc_size; i+=4){
    if(inp_pc[i+2] >= z_axis_min && inp_pc[i+2] <= z_axis_max){
      for(auto j=0; j<4; ++j)
        filtered_pc.push_back(inp_pc[i+j]);
    }
  }

  /*** Divide the point cloud into nested circular regions ***/
  //  This operation also takes 1 ms
  std::array<std::vector<int>, AdaptiveClustering::region_max_> indices_array;
  for(int i = 0; i < filtered_pc.size(); i+=4) {
    float range = 0.0;

    float d2 = pow(filtered_pc[i], 2) + pow(filtered_pc[i+1], 2) + pow(filtered_pc[i+2], 2);
    for(int j = 0; j < AdaptiveClustering::region_max_; j++) {
      if(d2 > range * range && d2 <= (range+regions_[j]) * (range+regions_[j])) {
        indices_array[j].push_back(i);
        break;
      }
      range += regions_[j];
    }
  }

  auto total_num_points = 0;
  for(int i = 0; i < AdaptiveClustering::region_max_; ++i){
    num_points_per_region[i] = indices_array[i].size();
    total_num_points += num_points_per_region[i];
  }

  inputEC_cat.clear();
  inputEC_cat.reserve(total_num_points * 4);
  for(auto& inds : indices_array){
    for(auto ind : inds){
      for(auto i=0; i<4; ++i)
        inputEC_cat.push_back(filtered_pc[ind+i]);
    }
  }

  std::vector<float*> inputEC_v(AdaptiveClustering::region_max_);
  std::vector<float*> outputEC_v(AdaptiveClustering::region_max_);
  std::vector<unsigned int*> indexEC_v(AdaptiveClustering::region_max_);
  std::vector<cudaStream_t> streams(AdaptiveClustering::region_max_);

  auto num_points_so_far = 0;
  for(int i = 0; i < AdaptiveClustering::region_max_; ++i){
    if(num_points_per_region[i] > cluster_size_min){
      cudaStreamCreate(&streams[i]);
      auto sz = sizeof(float) * 4 * num_points_per_region[i];
      cudaMallocManaged(&inputEC_v[i], sz, cudaMemAttachHost);
      cudaStreamAttachMemAsync(streams[i], inputEC_v[i]);
      cudaMallocManaged(&outputEC_v[i], sz, cudaMemAttachHost);
      cudaStreamAttachMemAsync(streams[i], outputEC_v[i]);
      cudaMallocManaged(&indexEC_v[i], sz / sizeof(float) * sizeof(unsigned int), cudaMemAttachHost);
      cudaStreamAttachMemAsync(streams[i], indexEC_v[i]);

      auto* data_ptr = inputEC_cat.data() + (num_points_so_far * 4);
      cudaMemcpyAsync(inputEC_v[i], data_ptr, sz, cudaMemcpyHostToDevice, streams[i]);
      cudaMemcpyAsync(outputEC_v[i], data_ptr, sz, cudaMemcpyHostToDevice, streams[i]);
    }
    num_points_so_far += num_points_per_region[i];
  }

  for(int r = 0; r < AdaptiveClustering::region_max_; r++) {
    if(num_points_per_region[r] > cluster_size_min) {
      cudaExtractCluster cudaec(streams[r]);
      cudaec.set(params[r]);
      cudaec.extract(inputEC_v[r], num_points_per_region[r], outputEC_v[r], indexEC_v[r]);
    }
  }

  std::vector<std::vector<float>> clusters;

  for(int r = 0; r < AdaptiveClustering::region_max_; r++) {
    if(num_points_per_region[r] > cluster_size_min) {
      cudaStreamSynchronize(streams[r]);
      cudaStreamDestroy(streams[r]);
      float *outputEC = outputEC_v[r];
      unsigned int *indexEC = indexEC_v[r];
      for(int i = 1; i <= indexEC[0]; i++) {
        std::vector<float> cloud_cluster;
        cloud_cluster.reserve(indexEC[i]*3);

        unsigned int outoff = 0;
        for(int w = 1; w < i; w++) {
          if(i > 1) {
            outoff += indexEC[w];
          }
        }

        for(std::size_t k = 0; k < indexEC[i]; ++k) {
          cloud_cluster.push_back(outputEC[(outoff+k)*4+0]);
          cloud_cluster.push_back(outputEC[(outoff+k)*4+1]);
          cloud_cluster.push_back(outputEC[(outoff+k)*4+2]);
        }

        clusters.push_back(std::move(cloud_cluster));
      }

      cudaFree(inputEC_v[r]);
      cudaFree(outputEC_v[r]);
      cudaFree(indexEC_v[r]);
    }
  }

  return clusters;
}
