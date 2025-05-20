FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
RUN mkdir -p /job/work /job/data /job/output

WORKDIR /job

# Copy necessary files to the appropriate locations
COPY model.py configurator.py train.py /job/

# Copy source files
COPY src/ /job/src/

# Copy model files from work directory
COPY work/meta.pkl /job/work/meta.pkl
COPY work/ckpt.pt /job/work/ckpt.pt

# Set up volumes
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Install dependencies
RUN pip install datasets
