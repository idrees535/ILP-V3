# Deploy the service on VastAI (Platform to rent GPU server)

Create https://vast.ai/ account

Register an SSH key from your local machine to your VastAI account

## Choose a server to rent on VastAI
Key notes:
- Use image: nvidia/cuda:11.8.0-devel-ubuntu22.04 tensorflow 2.14.0 works with cuda 11.
- At Docker Options: Enter `-p 8000:8000` because our django service uses this port. VastAI will then map this port to an open port on the host machine so that we can access from Internet.
- At Launch Mode, choose SSH


Install Github CLI
```
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
&& sudo mkdir -p -m 755 /etc/apt/keyrings \
&& wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
```

```
gh auth login

gh repo clone Tempest-Finance/ILP-Framework
```

Set up Ganache
```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
source ~/.bashrc
nvm install 16.3.0
nvm use 16.3.0
npm install -g ganache-cli@6.12.2
npm install -g solc
```

Install miniconda
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

```
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

Create conda with Python 3.9.19
```
conda create -n py3919 python=3.9.19 pip

conda activate py3919
```

Install requirements
```
cd ILP-framework
# If running on CPU only
pip install -r requirements_td.txt
# If GPU
pip install -r requirements_gpu.txt
```

Run the server
```
cd django_app
# 0.0.0.0 needs to be specified because it's a Docker thing to map back to parent localhost
RESET_ENV=1 python manage.py runserver 0.0.0.0:8000
```
