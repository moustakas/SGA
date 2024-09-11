FROM legacysurvey/legacypipe:DR10.3.1

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
  ; do pip --no-cache-dir install --upgrade $x; done

# Remove the policy.xml file so we do not get an 'exhausted cache resources'
# error when we build mosaics for very large systems.
RUN echo '<policymap></policymap>' > /etc/ImageMagick-6/policy.xml

ENV IPYTHONDIR /tmp/ipython-config

# SGA and legacyhalos
WORKDIR /src

RUN for x in \
  SGA \
  legacyhalos \
  ; do git clone https://github.com/moustakas/$x.git $x; done

RUN for x in \
  desiutil \
  ; do git clone https://github.com/desihub/$x.git $x; done

RUN git config --global --add safe.directory /src/SGA \
  && git config --global --add safe.directory /src/legacyhalos \
  && git config --global --add safe.directory /src/desiutil

# updates
RUN cd /src/SGA && git pull && echo 1 \
  && cd /src/legacyhalos && git pull && echo 1


ENV PYTHONPATH=/src/SGA/py:/src/legacyhalos/py:/src/desiutil/py:$PYTHONPATH
ENV PATH=/src/SGA/bin:/src/legacyhalos/bin:/src/desiutil/bin:$PATH

LABEL Maintainer="John Moustakas jmoustakas@siena.edu"
