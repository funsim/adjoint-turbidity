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

ENV IPOPT_VER=3.12.6
COPY dolfin-adjoint.conf $FENICS_HOME/dolfin-adjoint.conf

RUN apt-get install -y curl wget
RUN /bin/bash -l -c "source $FENICS_HOME/dolfin-adjoint.conf && \
                     update_ipopt && \
                     update_pyipopt"
RUN /bin/bash -l -c "source $FENICS_HOME/dolfin-adjoint.conf && \
                     update_libadjoint && \
                     update_dolfin-adjoint"

CMD ["bash"]
