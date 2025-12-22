FROM legacysurvey/legacypipe:gpu-1.4.16

RUN for x in \
  ipython \
  pydl \
  pyyaml \
  h5py \
  gnureadline \
  corner \
  scikit-image \
  scikit-learn \
  seaborn \
  photutils \
  healpy \
  astroquery \
  wurlitzer \
  ; do pip --no-cache-dir install --upgrade $x --break-system-packages; done

ENV IPYTHONDIR=/tmp/ipython-config

WORKDIR /src

RUN for x in \
  SGA \
  ; do git clone https://github.com/moustakas/$x.git $x; done

RUN for x in \
  desiutil \
  ; do git clone https://github.com/desihub/$x.git $x; done

RUN git config --global --add safe.directory /src/SGA \
  && git config --global --add safe.directory /src/desiutil

# updates
RUN cd /src/SGA && git pull && echo 4

ENV PYTHONPATH=/src/SGA/py:/src/desiutil/py:$PYTHONPATH

LABEL Maintainer="John Moustakas jmoustakas@siena.edu"
