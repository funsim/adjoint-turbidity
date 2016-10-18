# Builds a Docker image with OpenTidalFarm master
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
  rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y git

ENV HOME /root
WORKDIR /root

RUN git clone https://github.com/FluidityProject/fluidity.git && cd fluidity/libspud && ./configure && make install

RUN apt-get install -y curl wget unzip python-pip

ENV IPOPT_VER=3.12.6
COPY dolfin-adjoint.conf $FENICS_HOME/dolfin-adjoint.conf

RUN /bin/bash -l -c "source $FENICS_HOME/dolfin-adjoint.conf && \
                     update_ipopt && \
                     update_pyipopt && \
                     update_libadjoint && \
                     update_dolfin-adjoint"

RUN mkdir pgfplots && cd pgfplots && wget "http://downloads.sourceforge.net/project/pgfplots/pgfplots/1.14/pgfplots_1.14.tds.zip?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fpgfplots%2Ffiles%2Fpgfplots%2F1.14%2F&ts=1476811187&use_mirror=netix" -O pgfplots.zip && unzip pgfplots.zip
RUN echo "export PYTHONPATH=~/pgfplots/scripts/pgfplots" >> .bashrc

RUN /bin/bash -l -c "git clone https://simon_funke@bitbucket.org/simon_funke/adjoint-turbidity.git"
RUN echo "export PYTHONPATH=$PYTHONPATH:~/adjoint-turbidity" >> .bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/lib:~/fluidity/libspud/" >> .bashrc

WORKDIR /root/adjoint-turbidity
CMD ["bash"]
