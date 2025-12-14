# Menggunakan NVIDIA CUDA base image dengan Ubuntu
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# Install Python 3.13, git, dan git-lfs
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.13 sebagai default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1

# Menyalin binary 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv

WORKDIR /code

# Salin file definisi project
COPY pyproject.toml uv.lock ./

# Install dependencies dengan uv
RUN /uv sync --frozen --no-dev

# PENTING: Tambahkan environment uv ke PATH
ENV PATH="/code/.venv/bin:$PATH"

# Salin seluruh kode (termasuk .git folder agar LFS bisa jalan)
COPY ./ /code/

# Inisialisasi Git LFS dan tarik file asli
RUN git lfs install && git lfs pull || echo "Git LFS pull skipped or failed"

CMD ["fastapi", "run"]
