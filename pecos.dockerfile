# PECOS baseline runtime (libpecos has no macOS wheel). Used by test/xmr4el/pecos_run.py.
#   docker build --platform linux/amd64 -f pecos.dockerfile -t xmr4el-pecos .
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# CPU torch (libpecos depends on torch; the PyPI linux wheel pulls CUDA). PyPI stays as the extra index for
# torch's own deps: the CPU index alone serves mislabelled typing-extensions metadata that old pip rejects.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
 && pip install --no-cache-dir libpecos \
 && python -c "from pecos.xmc.xlinear.model import XLinearModel"
