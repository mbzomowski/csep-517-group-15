FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
RUN mkdir -p /job/data/flores /job/out-flores-eng /job/config

WORKDIR /job

# Copy necessary files to the appropriate locations
COPY model.py configurator.py train.py /job/
COPY config/train_flores_eng_char.py /job/config/

# Copy model files to multiple locations for redundancy
COPY data/flores/meta.pkl /job/data/flores/meta.pkl
COPY data/flores/meta.pkl /job/meta.pkl
COPY out-flores-eng/ckpt.pt /job/out-flores-eng/ckpt.pt
COPY out-flores-eng/ckpt.pt /job/ckpt.pt

# Copy source files
COPY src/ /job/src/

# Create necessary directories
RUN mkdir -p /job/work /job/output

# Set up volumes
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Install dependencies
RUN pip install datasets
