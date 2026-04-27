set -e
set -x

# Fix CentOS 7 mirror (vault.centos.org)
sed -i 's/mirror.centos.org/vault.centos.org/g' /etc/yum.repos.d/*.repo
sed -i 's/^#.*baseurl=http/baseurl=http/g' /etc/yum.repos.d/*.repo
sed -i 's/^mirrorlist=http/#mirrorlist=http/g' /etc/yum.repos.d/*.repo

# Add NVIDIA CUDA repo for RHEL9
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

# Install CUDA 12.6 toolkit (nvcc + cudart + driver headers)
yum install --setopt=obsoletes=0 -y \
    cuda-nvcc-12-6-12.6.77-1 \
    cuda-cudart-devel-12-6-12.6.77-1 \
    cuda-driver-devel-12-6-12.6.77-1

ln -sf /usr/local/cuda-12 /usr/local/cuda
