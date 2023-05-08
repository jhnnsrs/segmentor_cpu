FROM armswdev/tensorflow-arm-neoverse:r23.04-tf-2.11.0-eigen


RUN pip install stardist edt 
RUN pip install "arkitekt[cli]==0.4.109"

#RUN pip install grunnlag==0.4.5 s3fs==0.4.2 # 04.2 because its the last working s3fs for freeking python 3.6.9
#RUN pip install bergen==0.4.32

RUN mkdir /home/ubuntu/workspace
COPY . /home/ubuntu/workspace
WORKDIR /home/ubuntu/workspace
RUN sudo chown -R ubuntu:ubuntu /home/ubuntu/workspace