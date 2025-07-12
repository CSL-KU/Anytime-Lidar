FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0
ARG CUDA_ARCH
RUN /bin/bash -c 'apt-get update; \
        apt-get install -y vim tmux python-is-python3 htop ninja-build tree python3-pip libopenblas-dev git cmake; \
	python3 -m pip install --upgrade pip; \
	pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.3.0a0+6ddf5cf85e.nv24.04.14026654-cp310-cp310-linux_aarch64.whl; \
	wget https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl -O torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl; \
	pip install --no-cache torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl; \
        pip install -v pccm wheel SharedArray scikit-learn scikit-image av2==0.2.1 kornia==0.6.12 easydict tensorboardX alive-progress onnx onnxruntime numpy==1.26.4'

RUN /bin/bash -c 'cd; \
        git clone https://github.com/CSL-KU/Anytime-Lidar; \
        git clone -b v0.4.11 https://github.com/FindDefinition/cumm.git; \
        cd cumm; \
	git apply ../Anytime-Lidar/req_patches/cumm.patch; \
        MAX_JOBS=$(nproc) CUMM_CUDA_ARCH_LIST=$CUDA_ARCH CUMM_DISABLE_JIT="1" python setup.py bdist_wheel; \
        pip install dist/*'

RUN /bin/bash -c 'cd; \
        git clone -b v2.3.6 https://github.com/traveller59/spconv.git; \
        cd spconv; \
        MAX_JOBS=$(nproc) CUMM_CUDA_ARCH_LIST=$CUDA_ARCH SPCONV_DISABLE_JIT="1" python setup.py bdist_wheel; \
        pip install dist/*'

RUN /bin/bash -c 'cd; \
        git clone -b 2.1.1 https://github.com/rusty1s/pytorch_scatter.git; \
        cd pytorch_scatter; \
        CPATH=$CPATH:/usr/local/cuda/include TORCH_CUDA_ARCH_LIST=$CUDA_ARCH FORCE_CUDA=1 pip install -v .'

RUN /bin/bash -c 'cd; \
        git clone https://github.com/ahmedius2/sbnet.git; \
        cd sbnet/sbnet_pytorch/; \
        MAX_JOBS=$(nproc) TORCH_CUDA_ARCH_LIST=$CUDA_ARCH pip install -v .'

RUN /bin/bash -c 'cd; \
	echo "export PCDET_PATH=/root/Anytime-Lidar" >> ~/.bashrc; \
        cd Anytime-Lidar; \
        MAX_JOBS=$(nproc) TORCH_CUDA_ARCH_LIST=$CUDA_ARCH python setup.py develop'

RUN /bin/bash -c 'cd; \
	git clone -b release/8.6 https://github.com/NVIDIA/TensorRT.git'

RUN /bin/bash -c 'cd; \
	git clone -b anytime_lidar https://github.com/ahmedius2/nuscenes-devkit.git; \
	cd nuscenes-devkit; \
	pip install -r setup/requirements.txt; \
	pip install motmetrics==1.1.3 pyquaternion; \
	echo "export PYTHONPATH=\${PYTHONPATH}:\$HOME/nuscenes-devkit/python-sdk" >> ~/.bashrc; \
	echo "export PATH=\${PATH}:/usr/src/tensorrt/bin" >> ~/.bashrc'

RUN sed -i 's/from collections import OrderedDict, Iterable/from collections import OrderedDict\nfrom collections.abc import Iterable/' \
	'/usr/local/lib/python3.10/dist-packages/motmetrics/metrics.py'

RUN echo 'if [ -e "/var/lib/nvpmodel/status" ]; then' >> ~/.bashrc && \
    echo '    export PMODE=$(cat "/var/lib/nvpmodel/status" | tr '"'"':'"'"' '"'"'_'"'"')' >> ~/.bashrc && \
    echo 'else' >> ~/.bashrc && \
    echo '    export PMODE="UNKNOWN_POWER_MODE"' >> ~/.bashrc && \
    echo 'fi' >> ~/.bashrc

