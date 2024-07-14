FROM python:3.10

# replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

COPY . .

WORKDIR /django_app

RUN pip install "cython<3.0.0" wheel

RUN pip install "pyyaml==5.4.1" --no-build-isolation

RUN pip install -r ../requirements_td.txt

ENV RESET_ENV 1

EXPOSE 8000

RUN chmod +x ./scripts/docker_run.sh

ENTRYPOINT ./scripts/docker_run.sh
