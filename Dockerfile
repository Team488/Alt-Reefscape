ARG TARGETPLATFORM
FROM johnylamw/488-alt-python-3.10
WORKDIR /xbot/Alt/src

RUN pip install --upgrade tensorflow
RUN pip install --upgrade XTablesClient
RUN pip install --upgrade ultralytics
RUN pip install --upgrade Flask


# ARG TARGETPLATFORM
# FROM johnylamw/alt-docker-base-python-3.10
# WORKDIR /xbot/Alt/src

# RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3-dev
#     # rm -rf /var/lib/apt/lists/* && apt-get clean

# # this dependency allows cmake to install
# RUN apt-get install -y python3-launchpadlib

# # Install cmake from the official repository
# RUN apt-get install -y software-properties-common && \
#     add-apt-repository ppa:george-edison55/cmake-3.x && \
#     apt-get update && \
#     apt-get install -y cmake


# RUN apt-get update && \
#     apt-get install -y python3-pip

# # installing robotpy__apriltag (currenntly this installs all of robotpy)
# WORKDIR /xbot/Alt
# RUN git clone https://github.com/robotpy/mostrobotpy.git
# # into repo
# WORKDIR /xbot/Alt/mostrobotpy
# # keep deterministic for mostrobotpy
# RUN git fetch origin
# RUN git checkout f16ab492127e01f8db152ecfd0de47acbce5674a

# RUN pip install pybind11
# RUN pip install --upgrade pip
# RUN pip install -r rdev_requirements.txt  # Install project-specific dependencies
# RUN pip install numpy  # Install numpy separately, as instructed
# RUN pip install devtools
# # Step 5: Make the rdev.sh script executable
# RUN chmod +x rdev.sh

# # Step 6: Run the build command to generate the wheels
# RUN ./rdev.sh ci run

# # Step 7: Install the resulting wheels
# RUN pip install dist/*.whl
# # to make this only install apriltag we can find that whl only, but it might break it


# # go back to regular workdir
# WORKDIR /xbot/Alt
# COPY non-base-requirements.txt /xbot/Alt/non-base-requirements.txt
# RUN pip install --no-cache-dir --prefer-binary -r non-base-requirements.txt

# COPY ./src/assets/librknnrt.so /usr/lib/librknnrt.so

# WORKDIR /xbot/Alt/src
# WORKDIR /xbot/Alt
# COPY non-base-requirements.txt /xbot/Alt/non-base-requirements.txt
# RUN pip install --no-cache-dir --prefer-binary -r non-base-requirements.txt

# COPY ./src/assets/librknnrt.so /usr/lib/librknnrt.so

# WORKDIR /xbot/Alt/src
