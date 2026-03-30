FROM python:3.12-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first to avoid pulling CUDA wheels (~2GB savings)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install ovos-tts-server and the plugin
COPY . /tmp/plugin
# Pin gradio-client to match gradio 3.x (ovos-tts-server pulls gradio 3.36)
RUN pip install --no-cache-dir "gradio-client<1.0" ovos-tts-server>=0.1.0 /tmp/plugin && \
    rm -rf /tmp/plugin

# Pre-download model weights so first request is fast
RUN python3 -c "\
from pocket_tts import TTSModel; \
model = TTSModel.load_model(); \
state = model.get_state_for_audio_prompt('marius'); \
print(f'Model loaded, sample_rate={model.sample_rate}')"

# Write default config — voice=marius, sample_rate=16000
RUN mkdir -p /root/.config/mycroft && \
    echo '{"tts": {"ovos-tts-plugin-pocket-tts": {"voice": "marius", "sample_rate": 16000}}}' \
    > /root/.config/mycroft/mycroft.conf

EXPOSE 9666

ENTRYPOINT ["ovos-tts-server", "--engine", "ovos-tts-plugin-pocket-tts", "--cache"]
