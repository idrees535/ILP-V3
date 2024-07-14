apt-get update
apt-get install -y curl
apt-get -y autoclean

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash

source ~/.bashrc
nvm install 16.3.0
nvm use 16.3.0

source ~/.bashrc
npm install -g ganache-cli
npm install -g solc

python manage.py runserver 0.0.0.0:8000