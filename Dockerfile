FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y --no-install-recommends\
	apt-utils \
	build-essential \
	cmake \
	git \
	curl \
	iputils-ping \
	python3 \
	python3-pip \
	python3-dev \
	python3-setuptools \
	python3-wheel

Run python3 -m pip install --upgrade pip

COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt

EXPOSE 8888

RUN mkdir /experiments
RUN chmod 777 /experiments

WORKDIR experiments
COPY src .


CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/experiments", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=localerror"]

