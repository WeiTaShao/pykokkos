# Load conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyk

# Force Kokkos + nvcc to use GCC 13 (CUDA 12.4 only supports GCC <= 13)
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDAHOSTCXX=/usr/bin/g++-13

# Force nvcc_wrapper to use g++-13 instead of /usr/bin/c++
export NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++-13
