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
    ; do pip --no-cache-dir install $x; done

# Remove the policy.xml file so we do not get an 'exhausted cache resources'
# error when we build mosaics for very large systems.
RUN echo '<policymap></policymap>' > /etc/ImageMagick-6/policy.xml

ENV IPYTHONDIR /tmp/ipython-config

# SGA and legacyhalos
WORKDIR /src

RUN git clone https://github.com/moustakas/SGA.git SGA \
    && cd SGA && git describe && echo 1
RUN git clone https://github.com/moustakas/legacyhalos.git legacyhalos \
    && cd legacyhalos && git describe && echo 1
RUN git config --global --add safe.directory /src/SGA
RUN git config --global --add safe.directory /src/legacyhalos

ENV PYTHONPATH=/src/SGA/py:/src/legacyhalos/py:$PYTHONPATH
ENV PATH=/src/SGA/bin:/src/legacyhalos/bin:$PATH

LABEL Maintainer="John Moustakas jmoustakas@siena.edu"
