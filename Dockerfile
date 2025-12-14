FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# Set timezone
ENV TZ=Asia/Jakarta
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependensi untuk Python 3.13
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    git \
    git-lfs \
    libssl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.13 sebagai default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1

WORKDIR /code

# Salin file definisi project
COPY pyproject.toml uv.lock ./

# Install dependencies dengan uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv
RUN /uv sync --frozen --no-dev

# Tambahkan environment uv ke PATH
ENV PATH="/code/.venv/bin:$PATH"

# Salin seluruh kode (tanpa .git)
COPY . /code/

# Inisialisasi Git LFS dan tarik file asli (jika diperlukan)
RUN if [ -f .gitmodules ] || git config --get-regexp 'filter.lfs' | grep -q 'clean'; then \
    git lfs install && git lfs pull; \
    else \
    echo "Git LFS not required, skipping..."; \
    fi

CMD ["fastapi", "run"]
