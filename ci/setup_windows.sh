set -e
set -x

CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
touch ~/.netrc
chmod 600 ~/.netrc
curl -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/12.8.0/network_installers/cuda_12.8.0_windows_network.exe
./cuda.exe -s nvcc_12.8 cudart_12.8
rm cuda.exe
