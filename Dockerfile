FROM    ubuntu:16.04

RUN     apt-get update \
        &&  apt-get install -y \
            build-essential \
            cmake \
            git \
            libgtk2.0-dev \
            pkg-config \
            libavcodec-dev \
            libavformat-dev \
            libswscale-dev  \

            libgoogle-glog-dev \
            libatlas-base-dev \
            libeigen3-dev \
            libsuitesparse-dev


RUN     git clone https://github.com/opencv/opencv.git \
        &&  cd opencv \
        &&  mkdir release \
        &&  cd release \
        &&  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. \
        &&  make -j3 \
        &&  make install


RUN     git clone https://ceres-solver.googlesource.com/ceres-solver \
RUN     cd ceres-solver \
        &&  mkdir release \
        &&  cd release

RUN       cmake -j8 ..
RUN       make -j8
RUN       make test
RUN       make install

RUN apt-get install -y apt-utils openssh-server gdb gdbserver rsync vim

RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# 22 for ssh server. 7777 for gdb server.
EXPOSE 22 7777

RUN useradd -ms /bin/bash debugger
RUN echo 'debugger:pwd' | chpasswd

CMD ["/usr/sbin/sshd", "-D"]
