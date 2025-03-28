import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources, extra_link_flags=None):
    if extra_link_flags is None:
        cuda_ext = CUDAExtension(
            name='%s.%s' % (module, name),
            sources=[os.path.join(*module.split('.'), src) for src in sources]
        )
    else:
        cuda_ext = CUDAExtension(
            name='%s.%s' % (module, name),
            sources=[os.path.join(*module.split('.'), src) for src in sources],
            extra_link_flags=extra_link_flags
        )

    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.6.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'pcdet/version.py')

    setup(
        name='pcdet',
        version=version,
        description='OpenPCDet is a general codebase for 3D object detection from point cloud',
        install_requires=[
            'numpy',
            'llvmlite',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml',
            'scikit-image',
            'tqdm',
            #'SharedArray',
            # 'spconv',  # spconv has different names depending on the cuda version
        ],

        author='Shaoshuai Shi',
        author_email='shaoshuaics@gmail.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='pcdet.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            CUDAExtension(
                name="pcdet.ops.forecasting.forecasting",
                sources=[
                    'pcdet/ops/forecasting/src/forecasting.cpp',
                    'pcdet/ops/forecasting/src/forecasting_impl.cu',
                    'pcdet/ops/iou3d_nms/src/iou3d_cpu.cpp'
                ],
                #extra_compile_args={'nvcc' : ['-Ipcdet/ops/iou3d_nms/src']},
                include_dirs=[os.path.realpath('pcdet/ops/iou3d_nms/src')],
                             #extra_link_flags=["-Lpcdet/ops/iou3d_nms",
                #     "-liou3d_nms_cuda.cpython-310-aarch64-linux-gnu"],
                #runtime_library_dirs=["pcdet/ops/iou3d_nm"]
            ),
            make_cuda_ext(
                name='dsvt_ops',
                module='pcdet.ops.dsvt_ops',
                sources=[
                    'src/dsvt_ops.cpp',
                    'src/dsvt_ops_cuda.cu',
                ]
            ),
            make_cuda_ext(
                name='ioubev_nms_cuda',
                module='pcdet.ops.ioubev_nms',
                sources=[
                    'src/iou3d.cpp',
                    'src/iou3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='pcdet.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roipoint_pool3d_cuda',
                module='pcdet.ops.roipoint_pool3d',
                sources=[
                    'src/roipoint_pool3d.cpp',
                    'src/roipoint_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='pointnet2_stack_cuda',
                module='pcdet.ops.pointnet2.pointnet2_stack',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu', 
                    'src/interpolate.cpp', 
                    'src/interpolate_gpu.cu',
                    'src/voxel_query.cpp', 
                    'src/voxel_query_gpu.cu',
                    'src/vector_pool.cpp',
                    'src/vector_pool_gpu.cu'
                ],
            ),
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='pcdet.ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',

                ],
            ),
            make_cuda_ext(
                name='cuda_slicer',
                module='pcdet.ops.cuda_slicer',
                sources=[
                    'src/slice_and_batch.cpp',
                    #'src/slice_and_batch_cuda.cu',
                    'src/slice_and_batch_nhwc_cuda.cu',
                ],
            ),
            make_cuda_ext(
                name='cuda_point_tile_mask',
                module='pcdet.ops.cuda_point_tile_mask',
                sources=[
                    'src/point_tile_mask.cpp',
                    'src/point_tile_mask_cuda.cu',
                ],
            ),

            make_cuda_ext(
                name='ingroup_inds_cuda',
                module='pcdet.ops.ingroup_inds',
                sources=[
                    'src/ingroup_inds.cpp',
                    'src/ingroup_inds_kernel.cu',
                ]
            ),
        ],
    )
