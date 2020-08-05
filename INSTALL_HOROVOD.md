# Horovod
concepts: https://horovod.readthedocs.io/en/latest/concepts_include.html
install: https://horovod.readthedocs.io/en/latest/gpus.html

### install prerequisites
```
sudo apt-get update
sudo apt-get install autoconf flex libtool # for autogen.pl
<!--sudo apt-get install libnccl2 libnccl-dev-->
<!--sudo apt-get install g++-4.9 libnvinfer5 #for tensorflow-->
```

### install nccl
https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html

### install open-mpi (compile 설치 권장)
Ubuntu 16.04에서 open-mpi가 1.10.x가 설치되어 제대로 사용할 수 없음
```
sudo apt-get remove openmpi-* libopenmpi1.*

cd
wget https://github.com/open-mpi/ompi/archive/v4.0.2.tar.gz
tar zxvf v4.0.2.tar.gz
cd ompi-4.0.2
./autogen.pl
./configure --prefix=/usr/local/openmpi
make && sudo make install

echo "" >>~/.profile
echo "# cuda" >>~/.profile
echo "export PATH=\$PATH:/usr/local/cuda/bin" >>~/.profile

echo "" >>~/.profile
echo "# open-mpi" >>~/.profile
echo "export PATH=\$PATH:/usr/local/openmpi/bin/" >>~/.profile
source ~/.profile

echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf
echo "/usr/local/openmpi/lib" > /etc/ld.so.conf.d/openmpi.conf
sudo ldconfig
ldconfig -p | grep -i libmpi
```

<!--### install horovod with tensorflow-->
<!--sudo apt-get install python3-libnvinfer=6.0.1-1+cuda10.2-->
<!--pip install tensorflow-gpu-->
<!--CC=/usr/bin/gcc-4.9 HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_MPI=1 pip install --no-cache-dir horovod-->

### install horovod with pytorch, mpi, nccl
```
pip install --upgrade pip
pip uninstall horovod
CC=/usr/bin/gcc-5 HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 pip install --no-cache-dir horovod
pip list | grep horovod
horovodrun --mpi -np 1 -H localhost:1 python -c 'import horovod.torch as hvd;hvd.init();print(hvd.mpi_built(), hvd.mpi_enabled())'
```

### bechmark horovod
```
cd
git clone https://github.com/horovod/horovod
cd examples
horovodrun --gloo -np 2 -H localhost:2 python pytorch_synthetic_benchmark.py
horovodrun --gloo -np 2 -H localhost:2 python pytorch_synthetic_benchmark.py --fp16-allreduce
horovodrun --mpi -np 2 -H localhost:2 python pytorch_synthetic_benchmark.py
horovodrun --mpi -np 2 -H localhost:2 python pytorch_synthetic_benchmark.py --fp16-allreduce
```
##### Results
- Model: resnet50
- Batch size: 32
- GPU: Titan XP (12GB)
- Number of GPUs: 2
```
```
| backend | options | Total img/sec on 2 GPU(s) | Allocated Memory (MB) |
|---|:---:|---:|---:|
|gloo| |395.6||
|gloo|--fp16-allreduce|408.8||
|mpi| |391.6||
|mpi|--fp16-allreduce|404.3||

# Etc
~/.profile
```
# ~/.profile: executed by the command interpreter for login shells.
# This file is not read by bash(1), if ~/.bash_profile or ~/.bash_login
# exists.
# see /usr/share/doc/bash/examples/startup-files for examples.
# the files are located in the bash-doc package.

# the default umask is set in /etc/profile; for setting the umask
# for ssh logins, install and configure the libpam-umask package.
#umask 022

# if running bash
if [ -n "$BASH_VERSION" ]; then
    # include .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
	. "$HOME/.bashrc"
    fi
fi

# set PATH so it includes user's private bin directories
PATH="$HOME/bin:$HOME/.local/bin:$PATH"
export LC_ALL=C

# by bage
alias cd_transformer='cd /home/svcapp/userdata/workspace/transformer-evolution-bage/transformer'
alias watch_nvidia='watch -n 3 nvidia-smi'
alias tail_train.log='tail -f train.log'

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# cuda
export PATH=$PATH:/usr/local/cuda/bin

# open-mpi
export PATH=$PATH:/usr/local/openmpi/bin/
```