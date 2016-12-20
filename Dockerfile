# Builds a Docker image with dolfin-adjoint master
# version built from gitsources. It is based on
# the dev-dolfin-adjoint image available at
#
#      quay.io/dolfinadjoint/dev-dolfin-adjoint
#
# Authors:
# Alberto Paganini <aldapa@gmail.com>

FROM ubuntu:14.04
MAINTAINER Simon Funke <simon@simula.no>


# Install.
RUN \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y build-essential && \
  apt-get install -y fenics && \
  apt-get install -y ipython && \
  apt-get install -y git && \
  apt-get install -y curl wget unzip python-pip && \
  apt-get install -y texlive-full && \
  rm -rf /var/lib/apt/lists/*

ENV HOME /root
WORKDIR /root

RUN git clone https://github.com/FluidityProject/fluidity.git && cd fluidity/libspud && ./configure && make install

ENV IPOPT_VER=3.12.6
COPY dolfin-adjoint.conf /root/dolfin-adjoint.conf

RUN /bin/bash -l -c "source /root/dolfin-adjoint.conf && \
                     update_ipopt && \
                     update_pyipopt && \
                     update_libadjoint && \
                     update_dolfin-adjoint"

RUN mkdir pgfplots && cd pgfplots && wget "http://downloads.sourceforge.net/project/pgfplots/pgfplots/1.14/pgfplots_1.14.tds.zip?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fpgfplots%2Ffiles%2Fpgfplots%2F1.14%2F&ts=1476811187&use_mirror=netix" -O pgfplots.zip && unzip pgfplots.zip
RUN echo "export PYTHONPATH=~/pgfplots/scripts/pgfplots" >> .bashrc

RUN /bin/bash -l -c "git clone https://simon_funke@bitbucket.org/simon_funke/adjoint-turbidity.git"
RUN echo "export PYTHONPATH=\$PYTHONPATH:~/adjoint-turbidity" >> .bashrc
RUN echo "export PYTHONPATH=\$PYTHONPATH:~/usr/local/lib/python2.7/dist-packages" >> .bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/lib:~/fluidity/libspud/" >> .bashrc

WORKDIR /root/adjoint-turbidity
CMD ["bash"]
