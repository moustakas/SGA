FROM legacysurvey/cutouts:v1.1

RUN for x in \
  ipython \
  pydl \
  pyyaml \
  h5py \
  gnureadline \
  corner \
  scikit-image \
  seaborn \
  photutils \
  healpy \
  astroquery \
  ; do pip --no-cache-dir install --upgrade $x --break-system-packages; done

# Remove the policy.xml file so we do not get an 'exhausted cache resources'
# error when we build mosaics for very large systems.
RUN echo '<policymap></policymap>' > /etc/ImageMagick-6/policy.xml

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
RUN cd /src/SGA && git pull && echo 2

ENV PYTHONPATH=/src/SGA/py:/src/desiutil/py:$PYTHONPATH
ENV PATH=/src/SGA/bin:/src/desiutil/bin:$PATH

LABEL Maintainer="John Moustakas jmoustakas@siena.edu"
