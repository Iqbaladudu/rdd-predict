FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# Set timezone
ENV TZ=Asia/Jakarta
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python 3.12 (lebih stabil)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    git \
    tensorrt \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 sebagai default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

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

CMD ["fastapi", "run"]
