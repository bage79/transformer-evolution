# 아래 lib을 먼저 설치해야, pyenv 설치와 pandas warning 이 발생하지 않음.
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev
sudo apt-get install liblzma-dev

# install pyenv on Ubuntu
curl https://pyenv.run | bash

# add this lines to .bash_profile or .profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >>~/.profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >>~/.profile
eval '"$(pyenv init -)"' >>~/.profile
eval '"$(pyenv virtualenv-init -)"' >>~/.profile

# install python on pyenv
pyenv install 3.7.5
pyenv rehash
pyenv virtualenv 3.7.5 transformer-evolution

# install python packages
cd transformer-evolution-bage
pyenv local transformer-evolution
pip install -r requirements.txt
