#include <string>
#include <vector>
#include <array>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>

#include "lib/cudaCluster.h"

class AdaptiveClustering{       // The class
  public:             // Access specifier
    AdaptiveClustering(std::string sensor_model);
    ~AdaptiveClustering();

    std::vector<std::vector<float>> cluster(
        const float* inp_pc, const unsigned inp_pc_size); // input point cloud, x y z i format
  private:
    static const int region_max_ = 10; // Change this value to match how far you want to detect.
    const std::string sensor_model_ = "HDL-32E";
    const float z_axis_min = -0.8;
    const float z_axis_max = 2.0;
    const int cluster_size_min = 3;
    const int cluster_size_max = 2200000;
    int regions_[100];

    //std::unique_ptr<cudaExtractCluster> cudaec_ptr;
    std::vector<extractClusterParam_t> params;
    std::array<unsigned int, AdaptiveClustering::region_max_> num_points_per_region;
    std::vector<float> inputEC_cat;
    float *inputEC_all = NULL;
    float *outputEC_all = NULL;
    unsigned int *indexEC_all = NULL;

    //cudaStream_t stream = NULL;
};

