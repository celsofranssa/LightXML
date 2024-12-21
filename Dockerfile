FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt/project


RUN apt update
RUN apt install git -y
RUN apt install nano -y
RUN apt install python3.8-dev -y
RUN apt install python3-pip -y
RUN pip install packaging
RUN apt install ninja-build -y


# python packages
RUN pip install tokenizers==0.7.0
RUN pip install joblib==1.2.0
RUN pip install nltk==3.8.1
RUN pip install numpy==1.23.5
RUN pip install pandas==1.0.4
RUN pip install tqdm==4.46.1
RUN pip install scipy==1.10.1
RUN pip install transformers==2.11.0
RUN pip install torch~=1.13.1
RUN pip install scikit-learn~=0.23.2
RUN pip install ranx~=0.3.6
RUN python3 -m nltk.downloader punkt


RUN git clone https://github.com/NVIDIA/apex;
WORKDIR /opt/project/apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./;
WORKDIR /opt/project