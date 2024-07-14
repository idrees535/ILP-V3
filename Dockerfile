# FROM trufflesuite/ganache-cli:v6.12.2

# CMD ["ganache-cli", "--defaultBalanceEther", "10000000000000000000", "--gasLimit", "10000000000", "--gasPrice", "1", "--hardfork", "istanbul"]

FROM python:3.10

# replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# update the repository sources list
# and install dependencies
# RUN apt-get update \
#     && apt-get install -y curl \
#     && apt-get -y autoclean

# # install nvm
# # https://github.com/creationix/nvm#install-script
# RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash

# # install node
# RUN source ~/.bashrc \
#     && nvm install 16.3.0 \
#     && nvm use 16.3.0

# # install ganache-cli && solc
# RUN source ~/.bashrc \
#     && npm install -g ganache-cli \
#     && npm install -g solc

COPY . .

WORKDIR /django_app

RUN pip install "cython<3.0.0" wheel

RUN pip install "pyyaml==5.4.1" --no-build-isolation

RUN pip install -r ../requirements_td.txt

ENV RESET_ENV 1

EXPOSE 8000

RUN chmod +x ./scripts/docker_run.sh

ENTRYPOINT ./scripts/docker_run.sh

# CMD ["/bin/bash", "-c", "./scripts/docker_run.sh"]
