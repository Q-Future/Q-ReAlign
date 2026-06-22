# =============================================================================
# qalign — GPU runtime image.
#
# Built on the OFFICIAL, VERIFIED ModelScope + ms-swift image, which already
# ships the full heavy stack — torch 2.10, ms-swift 4.0.3, transformers,
# deepspeed, vLLM — on CUDA 12.8.1 / Python 3.11 / Ubuntu 22.04.
#
# We only layer qalign (+ decord for video) on top, so the image's pinned swift
# runtime is the source of truth (we deliberately do NOT `pip install .[runtime]`,
# which would try to pull ms-swift==4.0.2 and fight the base image).
#
#   docker build -t qalign .
#   docker run --gpus all -it --rm -v "$PWD":/workspace/qalign qalign
#
# Region mirrors: override BASE, e.g.
#   --build-arg BASE=modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3
# =============================================================================
ARG BASE=modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3
FROM ${BASE}

WORKDIR /workspace/qalign
COPY . /workspace/qalign

# Core deps (numpy/scipy/pillow/pyyaml) + the package, editable. decord for video.
# torch / ms-swift / transformers / deepspeed come from the base image untouched.
RUN pip install --no-cache-dir -e . decord \
    && python -c "import qalign; print('qalign', qalign.__version__, 'OK')"

CMD ["bash"]
