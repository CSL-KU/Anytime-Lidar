#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

//#include "pcdet/ops/iou3d_nms/src/iou3d_cpu.h"
#include "iou3d_cpu.h"

// Using double instead of float doesn't increase
// accuracy as far as I have tested
//#define USE_DOUBLE

// check https://discuss.pytorch.org/t/using-at-parallel-for-in-a-custom-operator/82747/4
// if you want to make cpu op parallel

#ifdef USE_DOUBLE
using fp_type = double;
#define FABS(x) fabs(x)
#define SQRT(x) sqrt(x)
#define COS(x) cos(x)
#define SIN(x) sin(x)
#define ATAN2(x,y) atan2(x,y)
#else
using fp_type = float;
#define FABS(x) fabsf(x)
#define SQRT(x) sqrtf(x)
#define COS(x) cosf(x)
#define SIN(x) sinf(x)
#define ATAN2(x,y) atan2f(x,y)
#endif

template <typename scalar_t>
using one_dim_pa32 = torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits>;

template <typename scalar_t>
using two_dim_pa32 = torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits>;

template <typename scalar_t>
using one_dim_acc = torch::TensorAccessor<scalar_t,1>;

template <typename scalar_t>
using two_dim_acc = torch::TensorAccessor<scalar_t,2>;

class Quaternion{
    public:
        __host__ __device__ Quaternion(fp_type r, fp_type i, fp_type j, fp_type k){
            r_ = r;
            i_ = i;
            j_ = j;
            k_ = k;
        }

        __host__ __device__ Quaternion(fp_type *axis, fp_type angle_rad){
            fp_type axis_x = axis[0];
            fp_type axis_y = axis[1];
            fp_type axis_z = axis[2];
            fp_type mag_sq = axis_x*axis_x + axis_y+axis_y + axis_z*axis_z;
            if(FABS(1.0f - mag_sq) > 1e-12){
                fp_type s = SQRT(mag_sq);
                axis_x /= s;
                axis_y /= s;
                axis_z /= s;
            }
            fp_type theta = angle_rad / 2.0;
            r_ = COS(theta);
            fp_type st = SIN(theta);
            i_ = axis_x * st;
            j_ = axis_y * st;
            k_ = axis_z * st;
        }

        __host__ __device__ void rmul_inplace(Quaternion &q){
            // multiply the q matrix of q with self
            fp_type r_tmp = q.r_*r_ - q.i_*i_ - q.j_*j_ - q.k_*k_;
            fp_type i_tmp = q.i_*r_ + q.r_*i_ - q.k_*j_ + q.j_*k_;
            fp_type j_tmp = q.j_*r_ + q.k_*i_ + q.r_*j_ - q.i_*k_;
            k_ = q.k_*r_ - q.j_*i_ + q.i_*j_ + q.r_*k_;
            j_ = j_tmp;
            i_ = i_tmp;
            r_ = r_tmp;
        }

        __host__ __device__ void invert_inplace(){
            fp_type ss = sum_of_squares();
            r_ = r_ / ss;
            i_ = -i_ / ss;
            j_ = -j_ / ss;
            k_ = -k_ / ss;
        }

        __host__ __device__ bool is_unit(fp_type tolarance = 1e-14) const{
            return (FABS(1.0) - sum_of_squares()) < tolarance;
        }

        __host__ __device__ fp_type sum_of_squares() const {
            return r_*r_ + i_*i_ + j_*j_ + k_*k_;
        }

        __host__ __device__ void normalise() {
            if(!is_unit()){
                fp_type n = SQRT(sum_of_squares());
                r_ /= n;
                i_ /= n;
                j_ /= n;
                k_ /= n;
            }
        }

        __host__ __device__ fp_type* rot_matrix() {
            normalise();
            calc_rot_matrix();
            return &rot_matrix_[0];
        }

        __host__ __device__ fp_type r() const {return r_;}
        __host__ __device__ fp_type i() const {return i_;}
        __host__ __device__ fp_type j() const {return j_;}
        __host__ __device__ fp_type k() const {return k_;}

        __host__ __device__ void print(char* pre_str){
            printf("%s\nr i j k: %f %f %f %f\n", pre_str, r_, i_, j_, k_);
            calc_rot_matrix();
            printf("Rot matrix:\n%f %f %f\n%f %f %f\n%f %f %f\n",
                    rot_matrix_[0], rot_matrix_[1], rot_matrix_[2],
                    rot_matrix_[3], rot_matrix_[4], rot_matrix_[5],
                    rot_matrix_[6], rot_matrix_[7], rot_matrix_[8]);
        }

        // NOTE, the default copy constructor of this class might cause problems

    private:
        __host__ __device__ void calc_rot_matrix(){
            // calc rotation matrix, doing matrix mult inplace
            fp_type r2 = r_*r_;
            fp_type i2 = i_*i_;
            fp_type j2 = j_*j_;
            fp_type k2 = k_*k_;
            fp_type ij = i_*j_;
            fp_type rk = r_*k_;
            fp_type ik = i_*k_;
            fp_type rj = r_*j_;
            fp_type jk = j_*k_;
            fp_type ri = r_*i_;

            rot_matrix_[0] = i2 + r2 - k2 - j2;
            rot_matrix_[1] = ij - rk - rk + ij;
            rot_matrix_[2] = ik + rj + ik + rj;
            rot_matrix_[3] = ij + rk + rk + ij;
            rot_matrix_[4] = j2 - k2 + r2 - i2;
            rot_matrix_[5] = jk + jk - ri - ri;
            rot_matrix_[6] = ik - rj + ik - rj;
            rot_matrix_[7] = jk + jk + ri + ri;
            rot_matrix_[8] = k2 - j2 - i2 + r2;
        }

        fp_type r_, i_, j_, k_;
        fp_type rot_matrix_[9]; // 3x3
};

class Box{
    public:
        __host__ __device__ Box(fp_type center_x, fp_type center_y, fp_type center_z,
                fp_type size_x, fp_type size_y, fp_type size_z,
                Quaternion &q, fp_type vel_x, fp_type vel_y, fp_type vel_z) :
            cx(center_x), cy(center_y), cz(center_z),
            sx(size_x), sy(size_y), sz(size_z),
            orientation(q), vx(vel_x), vy(vel_y), vz(vel_z) { }

        __host__ __device__ fp_type center_x() const { return cx;}
        __host__ __device__ fp_type center_y() const { return cy;}
        __host__ __device__ fp_type center_z() const { return cz;}
        __host__ __device__ fp_type size_x() const { return sx;}
        __host__ __device__ fp_type size_y() const { return sy;}
        __host__ __device__ fp_type size_z() const { return sz;}
        __host__ __device__ fp_type vel_x() const { return vx;}
        __host__ __device__ fp_type vel_y() const { return vy;}
        __host__ __device__ fp_type vel_z() const { return vz;}
        __host__ __device__ fp_type r() const { return orientation.r(); }
        __host__ __device__ fp_type i() const { return orientation.i(); }
        __host__ __device__ fp_type j() const { return orientation.j(); }
        __host__ __device__ fp_type k() const { return orientation.k(); }

        __host__ __device__ void translate(fp_type x, fp_type y, fp_type z){
            cx += x;
            cy += y;
            cz += z;
        }

        __host__ __device__ void rotate(Quaternion& q){
            fp_type* rm = q.rot_matrix();
            fp_type cx_tmp = rm[0]*cx + rm[1]*cy + rm[2]*cz;
            fp_type cy_tmp = rm[3]*cx + rm[4]*cy + rm[5]*cz;
            cz = rm[6]*cx + rm[7]*cy + rm[8]*cz;
            cy = cy_tmp;
            cx = cx_tmp;

            orientation.rmul_inplace(q);

            fp_type vx_tmp = rm[0]*vx + rm[1]*vy + rm[2]*vz;
            fp_type vy_tmp = rm[3]*vx + rm[4]*vy + rm[5]*vz;
            vz = rm[6]*vx + rm[7]*vy + rm[8]*vz;
            vy = vy_tmp;
            vx = vx_tmp;
        } 

        __host__ __device__ void print(char *pre_str){
            printf("%s\n"
                    "Center:      %f %f %f\n"
                    "Size:        %f %f %f\n"
                    "Velocity:    %f %f %f\n"
                    "Orientation: %f %f %f %f\n",
                    pre_str, cx, cy, cz, sx, sy, sz, vx, vy, vz, r(), i(), j(), k());
        }
    private:
        fp_type cx, cy, cz, sx, sy, sz, vx, vy, vz;
        Quaternion orientation;
};

__global__ void forecast_cuda_kernel(
        const two_dim_pa32<fp_type>   pred_boxes,
        const one_dim_pa32<long>  past_pose_indexes,
        const two_dim_pa32<fp_type>   past_poses,
        const one_dim_pa32<fp_type>   cur_pose,
        const one_dim_pa32<long>      past_ts,
        const long                    cur_ts,
        two_dim_pa32<fp_type>         forecasted_boxes) {
    // blockIdx.x is the block id
    // blockDim.x is the number of threads in a block
    // threadIdx.x is the thread id in the block
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < pred_boxes.size(0)){
        fp_type axis[3] = {0., 0., 1.};
        Quaternion q(axis, pred_boxes[idx][6]);
        Box pred_box(pred_boxes[idx][0], pred_boxes[idx][1], pred_boxes[idx][2],
                pred_boxes[idx][3], pred_boxes[idx][4], pred_boxes[idx][5],
                q, pred_boxes[idx][7], pred_boxes[idx][8], 0);
        auto pose_idx = past_pose_indexes[idx];

        Quaternion csr_q(past_poses[pose_idx][3], past_poses[pose_idx][4],
                past_poses[pose_idx][5], past_poses[pose_idx][6]);
        pred_box.rotate(csr_q);

        pred_box.translate(past_poses[pose_idx][0], past_poses[pose_idx][1],
                past_poses[pose_idx][2]);

        Quaternion epr_q(past_poses[pose_idx][10], past_poses[pose_idx][11],
                past_poses[pose_idx][12], past_poses[pose_idx][13]);
        pred_box.rotate(epr_q);

        pred_box.translate(past_poses[pose_idx][7], past_poses[pose_idx][8],
                past_poses[pose_idx][9]);

        fp_type elapsed_sec = (fp_type)(cur_ts - past_ts[pose_idx]) / 1000000.0;
        fp_type x_diff = pred_box.vel_x()*elapsed_sec;
        fp_type y_diff = pred_box.vel_y()*elapsed_sec;
        if (isfinite(x_diff) && isfinite(y_diff))
            pred_box.translate(x_diff, y_diff, 0);

        // Now use cure pose but inverted
        pred_box.translate(-cur_pose[7], -cur_pose[8], -cur_pose[9]);

        Quaternion epr_inv_q(cur_pose[10], cur_pose[11], cur_pose[12], cur_pose[13]);
        epr_inv_q.invert_inplace();
        pred_box.rotate(epr_inv_q);

        pred_box.translate(-cur_pose[0], -cur_pose[1], -cur_pose[2]);

        Quaternion csr_inv_q(cur_pose[3], cur_pose[4], cur_pose[5], cur_pose[6]);
        csr_inv_q.invert_inplace();
        pred_box.rotate(csr_inv_q);

        forecasted_boxes[idx][0] = pred_box.center_x();
        forecasted_boxes[idx][1] = pred_box.center_y();
        forecasted_boxes[idx][2] = pred_box.center_z();
        forecasted_boxes[idx][3] = pred_box.size_x();
        forecasted_boxes[idx][4] = pred_box.size_y();
        forecasted_boxes[idx][5] = pred_box.size_z();

        fp_type r = pred_box.r();
        fp_type i = pred_box.i();
        fp_type j = pred_box.j();
        fp_type k = pred_box.k();

        forecasted_boxes[idx][6] = 2 * ATAN2(SQRT(i*i+j*j+k*k), r);
        forecasted_boxes[idx][7] = pred_box.vel_x();
        forecasted_boxes[idx][8] = pred_box.vel_y();
    }
}

void forecast_cpu_kernel(
        const two_dim_acc<fp_type>   pred_boxes,
        const one_dim_acc<long>      past_pose_indexes,
        const two_dim_acc<fp_type>   past_poses,
        const one_dim_acc<fp_type>   cur_pose,
        const one_dim_acc<long>      past_ts,
        const long                   cur_ts,
        two_dim_acc<fp_type>        forecasted_boxes) {
    // blockIdx.x is the block id
    // blockDim.x is the number of threads in a block
    // threadIdx.x is the thread id in the block
    for(auto idx=0; idx < pred_boxes.size(0); ++idx){
        fp_type axis[3] = {0., 0., 1.};
        Quaternion q(axis, pred_boxes[idx][6]);
        Box pred_box(pred_boxes[idx][0], pred_boxes[idx][1], pred_boxes[idx][2],
                pred_boxes[idx][3], pred_boxes[idx][4], pred_boxes[idx][5],
                q, pred_boxes[idx][7], pred_boxes[idx][8], 0);
        auto pose_idx = past_pose_indexes[idx];

        Quaternion csr_q(past_poses[pose_idx][3], past_poses[pose_idx][4],
                past_poses[pose_idx][5], past_poses[pose_idx][6]);
        pred_box.rotate(csr_q);

        pred_box.translate(past_poses[pose_idx][0], past_poses[pose_idx][1],
                past_poses[pose_idx][2]);

        Quaternion epr_q(past_poses[pose_idx][10], past_poses[pose_idx][11],
                past_poses[pose_idx][12], past_poses[pose_idx][13]);
        pred_box.rotate(epr_q);

        pred_box.translate(past_poses[pose_idx][7], past_poses[pose_idx][8],
                past_poses[pose_idx][9]);

        fp_type elapsed_sec = (fp_type)(cur_ts - past_ts[pose_idx]) / 1000000.0;
        fp_type x_diff = pred_box.vel_x()*elapsed_sec;
        fp_type y_diff = pred_box.vel_y()*elapsed_sec;
        if (isfinite(x_diff) && isfinite(y_diff))
            pred_box.translate(x_diff, y_diff, 0);

        // Now use cure pose but inverted
        pred_box.translate(-cur_pose[7], -cur_pose[8], -cur_pose[9]);

        Quaternion epr_inv_q(cur_pose[10], cur_pose[11], cur_pose[12], cur_pose[13]);
        epr_inv_q.invert_inplace();
        pred_box.rotate(epr_inv_q);

        pred_box.translate(-cur_pose[0], -cur_pose[1], -cur_pose[2]);

        Quaternion csr_inv_q(cur_pose[3], cur_pose[4], cur_pose[5], cur_pose[6]);
        csr_inv_q.invert_inplace();
        pred_box.rotate(csr_inv_q);

        forecasted_boxes[idx][0] = pred_box.center_x();
        forecasted_boxes[idx][1] = pred_box.center_y();
        forecasted_boxes[idx][2] = pred_box.center_z();
        forecasted_boxes[idx][3] = pred_box.size_x();
        forecasted_boxes[idx][4] = pred_box.size_y();
        forecasted_boxes[idx][5] = pred_box.size_z();

        fp_type r = pred_box.r();
        fp_type i = pred_box.i();
        fp_type j = pred_box.j();
        fp_type k = pred_box.k();

        forecasted_boxes[idx][6] = 2 * ATAN2(SQRT(i*i+j*j+k*k), r);
        forecasted_boxes[idx][7] = pred_box.vel_x();
        forecasted_boxes[idx][8] = pred_box.vel_y();
    }
}


// This will be on cpu
void move_to_world_coords_kernel(
        const two_dim_acc<fp_type> pred_boxes,
        const two_dim_acc<fp_type> poses,
        const one_dim_acc<long>    pose_idx,
        two_dim_acc<fp_type>       moved_boxes) {

    for(auto idx=0; idx < pred_boxes.size(0); ++idx){
        fp_type axis[3] = {0., 0., 1.};
        Quaternion q(axis, pred_boxes[idx][6]);
        Box pred_box(pred_boxes[idx][0], pred_boxes[idx][1], pred_boxes[idx][2],
                pred_boxes[idx][3], pred_boxes[idx][4], pred_boxes[idx][5],
                q, pred_boxes[idx][7], pred_boxes[idx][8], 0);

        auto pose_idx_ = pose_idx[idx];

        Quaternion csr_q(poses[pose_idx_][3], poses[pose_idx_][4], poses[pose_idx_][5], poses[pose_idx_][6]);
        pred_box.rotate(csr_q);

        pred_box.translate(poses[pose_idx_][0], poses[pose_idx_][1],poses[pose_idx_][2]);

        Quaternion epr_q(poses[pose_idx_][10], poses[pose_idx_][11],poses[pose_idx_][12], poses[pose_idx_][13]);
        pred_box.rotate(epr_q);

        pred_box.translate(poses[pose_idx_][7], poses[pose_idx_][8],poses[pose_idx_][9]);

        moved_boxes[idx][0] = pred_box.center_x();
        moved_boxes[idx][1] = pred_box.center_y();
        moved_boxes[idx][2] = pred_box.center_z();
        moved_boxes[idx][3] = pred_box.size_x();
        moved_boxes[idx][4] = pred_box.size_y();
        moved_boxes[idx][5] = pred_box.size_z();

        fp_type r = pred_box.r();
        fp_type i = pred_box.i();
        fp_type j = pred_box.j();
        fp_type k = pred_box.k();

        moved_boxes[idx][6] = 2 * ATAN2(SQRT(i*i+j*j+k*k), r);
        moved_boxes[idx][7] = pred_box.vel_x();
        moved_boxes[idx][8] = pred_box.vel_y();
    }
}

torch::Tensor forecast_past_dets(
        const torch::Tensor pred_boxes, // [num_objects, 9], fp_type
        const torch::Tensor past_pose_indexes, // [num_objects], long
        const torch::Tensor past_poses, // [num_past_poses, 14], fp_type
        const torch::Tensor cur_pose, // [14], fp_type
        const torch::Tensor past_timestamps, // [num_past_poses], long
        const long target_timestamp // [1]
        )
{
    torch::Tensor forecasted_boxes = torch::empty_like(pred_boxes);

    if (pred_boxes.device().type() == torch::kCPU){
        forecast_cpu_kernel(
                pred_boxes.accessor<fp_type,2>(),
                past_pose_indexes.accessor<long,1>(),
                past_poses.accessor<fp_type,2>(),
                cur_pose.accessor<fp_type,1>(),
                past_timestamps.accessor<long,1>(),
                target_timestamp,
                forecasted_boxes.accessor<fp_type,2>());
    }
    else{
        const auto threads_per_block = 256;
        const auto num_blocks = std::ceil((fp_type)pred_boxes.size(0) / threads_per_block);
        auto s = at::cuda::getCurrentCUDAStream();
        forecast_cuda_kernel<<<num_blocks, threads_per_block, 0, s>>>(
                pred_boxes.packed_accessor32<fp_type,2,torch::RestrictPtrTraits>(),
                past_pose_indexes.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                past_poses.packed_accessor32<fp_type,2,torch::RestrictPtrTraits>(),
                cur_pose.packed_accessor32<fp_type,1,torch::RestrictPtrTraits>(),
                past_timestamps.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                target_timestamp,
                forecasted_boxes.packed_accessor32<fp_type,2,torch::RestrictPtrTraits>());
    }

    return forecasted_boxes;
}


torch::Tensor move_to_world_coords(
        const torch::Tensor pred_boxes, // [num_objects, 9], fp_type
        const torch::Tensor poses, // [num_objects, 9], fp_type
        const torch::Tensor pose_idx // [14], fp_type
        )
{
    torch::Tensor moved_boxes = torch::empty_like(pred_boxes);

    move_to_world_coords_kernel(pred_boxes.accessor<fp_type, 2>(),
            poses.accessor<fp_type, 2>(),
            pose_idx.accessor<long, 1>(),
            moved_boxes.accessor<fp_type, 2>());

    return moved_boxes;
}

// hp: high priority, lp: low priority
torch::Tensor forecasting_nms(
        const torch::Tensor pred_boxes_hp,
        const torch::Tensor pred_labels_hp,
        const torch::Tensor pred_boxes_lp,
        const torch::Tensor pred_labels_lp,
        const double iou_threshold)
{
    using namespace torch::indexing;

    auto tensor_options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU)
        .requires_grad(false);

    torch::Tensor ans_iou = torch::zeros({pred_boxes_hp.size(0), pred_boxes_lp.size(0)},
            tensor_options);
    auto pb_hp = pred_boxes_hp.index({Slice(), Slice(None, 7)}).contiguous();
    auto pb_lp = pred_boxes_lp.index({Slice(), Slice(None, 7)}).contiguous();
    boxes_iou_bev_with_labels_cpu(pb_hp, pred_labels_hp, pb_lp, pred_labels_lp, ans_iou);

    torch::Tensor keep_mask = torch::ones(pred_boxes_lp.size(0), tensor_options.dtype(torch::kBool));

    auto iou_a = ans_iou.accessor<float, 2>();
    auto lbl_hp = pred_labels_hp.accessor<long, 1>();
    auto lbl_lp = pred_labels_lp.accessor<long, 1>();
    
    for(auto i=0; i<iou_a.size(0); ++i){
        for(auto j=0; j<iou_a.size(1); ++j){
            if(iou_a[i][j] >= iou_threshold){
                keep_mask[j] = false;
            }
        }
    }
    
    return keep_mask;
}


std::vector<std::map<std::string, torch::Tensor>> split_dets(
        const torch::Tensor pred_boxes, // [num_objects, 9], fp_type
        const torch::Tensor pred_scores,
        const torch::Tensor pred_labels,
        const torch::Tensor cls_id_to_det_head_idx_map,
        const long num_det_heads,
        const bool move_to_gpu)
{
    using namespace torch::indexing;
    torch::Tensor det_head_mappings = cls_id_to_det_head_idx_map.index({pred_labels});

    std::vector<std::map<std::string, torch::Tensor>> forc_dicts(num_det_heads);
    auto pred_merged = torch::cat({pred_boxes, pred_scores.unsqueeze(-1),
            pred_labels.to(torch::kFloat32).unsqueeze(-1)}, -1);
    for(auto i=0; i<num_det_heads; ++i){
        std::map<std::string, torch::Tensor> pdict;

        auto pred_masked = pred_merged.index({(det_head_mappings == i)});
        auto pb = pred_masked.index({Slice(), Slice(None, -2)});
        auto ps = pred_masked.index({Slice(), -2});
        auto pl = pred_masked.index({Slice(), -1}).to(torch::kLong);
        pdict["pred_boxes"] = (move_to_gpu ? pb.to(torch::kCUDA) : pb);
        pdict["pred_scores"] = (move_to_gpu ? ps.to(torch::kCUDA) : ps);
        pdict["pred_labels"] = (move_to_gpu ? pl.to(torch::kCUDA) : pl);
        forc_dicts[i] = std::move(pdict);
    }
    return forc_dicts;
}
