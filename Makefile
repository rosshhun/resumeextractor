# Makefile for Skill Extractor Project
# Python and virtualenv
PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin

# CUDA version to install if needed
CUDA_VERSION := 11.8

# Check if GPU is available
GPU_AVAILABLE := $(shell lspci | grep -i nvidia)

# PHONY targets
.PHONY: all check-gpu setup-cuda create-venv install-deps download-nltk-data clean

all: check-gpu create-venv install-deps download-nltk-data

# Check for GPU and install CUDA if necessary
check-gpu:
ifneq ($(GPU_AVAILABLE),)
	@echo "GPU detected. Checking CUDA installation..."
	@if ! command -v nvcc &> /dev/null; then \
		echo "CUDA not found. Installing CUDA $(CUDA_VERSION)..."; \
		wget https://developer.download.nvidia.com/compute/cuda/$(CUDA_VERSION)/local_installers/cuda_$(CUDA_VERSION)_linux.run; \
		sudo sh cuda_$(CUDA_VERSION)_linux.run --silent --toolkit; \
		rm cuda_$(CUDA_VERSION)_linux.run; \
		echo 'export PATH=$$PATH:/usr/local/cuda-$(CUDA_VERSION)/bin' >> ~/.bashrc; \
		echo 'export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/usr/local/cuda-$(CUDA_VERSION)/lib64' >> ~/.bashrc; \
	else \
		echo "CUDA is already installed."; \
	fi
else
	@echo "No GPU detected. Skipping CUDA installation."
endif

# Create virtual environment
create-venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

# Install dependencies
install-deps: create-venv
	@echo "Installing dependencies..."
	$(BIN)/pip install -r requirements.txt
ifneq ($(GPU_AVAILABLE),)
	@echo "Installing PyTorch with CUDA support..."
	$(BIN)/pip install torch torchvision torchaudio
else
	@echo "Installing PyTorch without CUDA support..."
	$(BIN)/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
endif

# Download NLTK data
download-nltk-data: create-venv
	@echo "Downloading NLTK data..."
	$(BIN)/python -m nltk.downloader punkt stopwords wordnet

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

# Run the main script
run: all
	@echo "Running the main script..."
	$(BIN)/python main.py

# Setup everything and run
setup-and-run: all run