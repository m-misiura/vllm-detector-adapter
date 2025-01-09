# Use the specified base image
# FROM registry.access.redhat.com/ubi9/python-311 as base
FROM registry.access.redhat.com/ubi9/python-312 as base
# Switch to root temporarily to create the directory
USER 0

# Set environment variable for the shared package path
ENV SHARED_PACKAGE_PATH=/shared_packages/app

# Create the directory structure and set permissions
RUN mkdir -p ${SHARED_PACKAGE_PATH} && \
    chown -R 1001:0 /shared_packages && \
    chmod -R g+rwX /shared_packages

# Switch back to default user (1001)
USER 1001

# Set the working directory
WORKDIR /build

# Install Python and other dependencies
RUN pip install --upgrade --no-cache-dir pip wheel

# Copy only the necessary files for installation
COPY pyproject.toml .
COPY setup_requirements.txt .
COPY reqs.txt .
# Copy the main package directory
COPY vllm_detector_adapter/ vllm_detector_adapter/

# Install dependencies
RUN pip install -r setup_requirements.txt
RUN pip install -r reqs.txt

# Install the local package in development mode
RUN pip install -e .

# Copy all installed packages including our local package to the shared directory
RUN cp -r /opt/app-root/lib/python3.12/site-packages/* ${SHARED_PACKAGE_PATH}/ && \
    cp -r vllm_detector_adapter ${SHARED_PACKAGE_PATH}/vllm_detector_adapter/

WORKDIR /