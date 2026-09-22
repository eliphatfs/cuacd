set -e
set -x

# Fix CentOS 7 mirror (vault.centos.org)
sed -i 's/mirror.centos.org/vault.centos.org/g' /etc/yum.repos.d/*.repo
sed -i 's/^#.*baseurl=http/baseurl=http/g' /etc/yum.repos.d/*.repo
sed -i 's/^mirrorlist=http/#mirrorlist=http/g' /etc/yum.repos.d/*.repo

# Add NVIDIA CUDA repo for RHEL9 (CUDA 12.8 ships no RHEL7 packages, but the
# nvcc/ptxas binaries only need glibc >= 2.7, so they run fine on manylinux2014)
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

# Install CUDA 12.8 toolkit (nvcc + cudart headers + driver API headers/stubs)
yum install --setopt=obsoletes=0 -y \
    cuda-nvcc-12-8-12.8.61-1 \
    cuda-cudart-devel-12-8-12.8.57-1 \
    cuda-driver-devel-12-8-12.8.57-1

ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
