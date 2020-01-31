"""
https://horovod.readthedocs.io/en/latest/concepts_include.html
"""

# install open-mpi
#sudo apt-get install --reinstall openmpi-bin libopenmpi-dev # FIXME: Ubuntu16.04에서 설치 실패

# install horovod with mpi
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip install --no-cache-dir horovod
pip install --no-cache-dir horovod
echo "export PATH=\$PATH:/usr/lib/openmpi/lib/" >>~/.profile
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/openmpi/lib" >>~/.profile

# test horovod
cd
git clone https://github.com/horovod/horovod
cd examples
horovodrun --mpi -np 2 -H localhost:2 python pytorch_mnist.py
