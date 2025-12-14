# Menggunakan Python 3.13 versi slim (ringan)
FROM python:3.13-slim

# Install git dan git-lfs
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 git git-lfs

# Menyalin binary 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv

WORKDIR /code

# Salin file definisi project
COPY pyproject.toml uv.lock ./

# Install dependencies dengan uv
RUN /uv sync --frozen --no-dev

RUN export UV_TORCH_BACKEND=auto

# PENTING: Tambahkan environment uv ke PATH
ENV PATH="/code/.venv/bin:$PATH"

# Salin seluruh kode (termasuk .git folder agar LFS bisa jalan)
COPY ./ /code/

# --- TAMBAHAN BARU ---
# Inisialisasi Git LFS dan tarik file asli (mengganti pointer text dengan binary)
# Gunakan '|| true' agar build tidak error jika ini bukan repo git, 
# tapi pastikan file .pt anda sudah benar.
RUN git lfs install && git lfs pull || echo "Git LFS pull skipped or failed"
# ---------------------

CMD ["fastapi", "run"]