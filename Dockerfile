FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /job

# Copy necessary files to the appropriate locations
COPY model.py configurator.py train.py /job/

# Create necessary directories
RUN mkdir -p /job/work /job/output

# Copy model files to multiple locations for redundancy
COPY work/meta.pkl /job/meta.pkl
COPY work/ckpt.pt /job/ckpt.pt

# Copy source files
COPY src/ /job/src/

# Set up volumes
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Install dependencies
RUN pip install datasets
