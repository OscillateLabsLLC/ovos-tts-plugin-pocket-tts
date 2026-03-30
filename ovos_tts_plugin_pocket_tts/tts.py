"""Pocket TTS plugin for OVOS — lightweight, CPU-only TTS by Kyutai."""

import os
import struct
from typing import AsyncIterable

import numpy as np
from ovos_plugin_manager.templates.tts import StreamingTTS
from ovos_utils.log import LOG


# Lazy-loaded to keep import time < 500ms for opm-check
_model = None
_voice_states = {}

BUILTIN_VOICES = (
    "alba", "marius", "javert", "jean",
    "fantine", "cosette", "eponine", "azelma",
)

DEFAULT_VOICE = "alba"

# Sacrificial prefix to prevent first-word swallowing.
# Pocket-tts blends the voice prompt into the start of generation, which can
# eat the first word. Prepending "..." produces a short throwaway utterance
# followed by a silence gap that we detect and trim.
_PREFIX = "... "
_PREFIX_MIN_DURATION = 0.15  # seconds — don't look for gap before this
_PREFIX_MAX_DURATION = 1.0   # seconds — stop looking after this
_PREFIX_SILENCE_GAP = 0.08   # seconds — minimum silence to count as gap
_SILENCE_THRESHOLD = 0.01    # fraction of max amplitude


def _get_model():
    """Lazy-load and cache the Pocket TTS model."""
    global _model
    if _model is None:
        from pocket_tts import TTSModel

        LOG.info("Loading Pocket TTS model (first call — weights will download if needed)")
        _model = TTSModel.load_model()
        LOG.info("Pocket TTS model loaded")
    return _model


def _get_voice_state(model, voice: str):
    """Get or cache a voice state for the given voice identifier.

    voice can be:
      - A built-in voice name (e.g. "alba")
      - A path to a .wav file (voice cloning)
      - A path to a .safetensors file (exported voice state)
      - A HuggingFace URI (hf://kyutai/tts-voices/...)
    """
    if voice not in _voice_states:
        LOG.info(f"Loading voice state for: {voice}")
        _voice_states[voice] = model.get_state_for_audio_prompt(voice)
    return _voice_states[voice]


def _trim_prefix(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Remove the sacrificial prefix audio and leading/trailing silence."""
    if len(audio) == 0:
        return audio

    max_amp = np.abs(audio).max()
    if max_amp == 0:
        return audio

    threshold = max_amp * _SILENCE_THRESHOLD
    min_samples = int(sample_rate * _PREFIX_MIN_DURATION)
    max_samples = int(sample_rate * _PREFIX_MAX_DURATION)
    gap_samples = int(sample_rate * _PREFIX_SILENCE_GAP)

    prefix_end = 0
    if len(audio) > min_samples:
        search_end = min(len(audio), max_samples)
        is_silent = np.abs(audio[:search_end]) < threshold

        i = min_samples
        while i < search_end:
            if is_silent[i]:
                silence_start = i
                while i < search_end and is_silent[i]:
                    i += 1
                if (i - silence_start) >= gap_samples:
                    prefix_end = i
                    break
            else:
                i += 1

    if prefix_end > 0:
        LOG.debug(f"Trimmed prefix: {prefix_end} samples ({prefix_end / sample_rate:.3f}s)")
        audio = audio[prefix_end:]

    # Trim remaining leading silence (keep 50ms padding)
    non_silent = np.where(np.abs(audio) > threshold)[0]
    if len(non_silent) > 0:
        pad = int(sample_rate * 0.05)
        start = max(0, non_silent[0] - pad)
        end = min(len(audio), non_silent[-1] + pad + 1)
        audio = audio[start:end]

    return audio


def _resolve_voice(voice: str) -> str:
    """Normalize a voice identifier."""
    if not (os.path.sep in voice or voice.startswith("hf://") or voice.endswith((".wav", ".safetensors"))):
        return voice.lower()
    return voice


def _audio_to_int16(audio_np: np.ndarray) -> np.ndarray:
    """Clamp float audio to [-1, 1] and convert to int16 PCM."""
    return (np.clip(audio_np, -1, 1) * 32767).astype(np.int16)


def _resample(audio_np: np.ndarray, native_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio if rates differ."""
    if native_rate == target_rate:
        return audio_np
    from scipy.signal import resample

    num_samples = int(len(audio_np) * target_rate / native_rate)
    return resample(audio_np, num_samples).astype(np.float32)


def _make_wav_header(sample_rate: int, num_samples: int) -> bytes:
    """Build a 44-byte WAV header for mono 16-bit PCM."""
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", data_size,
    )


class PocketTTSPlugin(StreamingTTS):
    """Pocket TTS — 100M param, CPU-only, real-time TTS by Kyutai.

    Config example (mycroft.conf):
        "tts": {
            "module": "ovos-tts-plugin-pocket-tts",
            "ovos-tts-plugin-pocket-tts": {
                "voice": "alba",
                "enable_streaming": true
            }
        }

    The "voice" field accepts:
      - Built-in names: alba, marius, javert, jean, fantine, cosette, eponine, azelma
      - Path to a .wav file for voice cloning
      - Path to a .safetensors file for pre-exported voice states
      - HuggingFace URI: hf://kyutai/tts-voices/...
    """

    def __init__(self, config=None):
        super().__init__(config=config, audio_ext="wav")

    def get_tts(self, sentence: str, wav_file: str, lang: str = None, voice: str = None) -> tuple:
        """Synchronous TTS with prefix trimming and resampling."""
        import wave

        voice = _resolve_voice(voice or self.config.get("voice", DEFAULT_VOICE))
        model = _get_model()
        voice_state = _get_voice_state(model, voice)

        audio = model.generate_audio(voice_state, _PREFIX + sentence)
        audio_np = audio.clamp(-1, 1).numpy()
        audio_np = _trim_prefix(audio_np, model.sample_rate)

        target_rate = int(self.config.get("sample_rate", 16000))
        audio_np = _resample(audio_np, model.sample_rate, target_rate)
        audio_int16 = _audio_to_int16(audio_np)

        with wave.open(wav_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_rate)
            wf.writeframes(audio_int16.tobytes())

        return wav_file, None

    async def stream_tts(self, sentence: str, **kwargs) -> AsyncIterable[bytes]:
        """Stream TTS audio chunks as they become available.

        Buffers the first ~1s of audio to trim the sacrificial prefix,
        then yields remaining chunks as raw PCM bytes wrapped in a WAV
        container.
        """
        voice = _resolve_voice(kwargs.get("voice") or self.config.get("voice", DEFAULT_VOICE))
        model = _get_model()
        voice_state = _get_voice_state(model, voice)
        native_rate = model.sample_rate
        target_rate = int(self.config.get("sample_rate", 16000))

        # Collect all chunks — we need the full audio for prefix trimming
        # and WAV header (which requires knowing total size).
        # Streaming benefit: model generates chunks incrementally so we
        # start processing sooner than generate_audio().
        all_chunks = []
        for chunk in model.generate_audio_stream(
            model_state=voice_state,
            text_to_generate=_PREFIX + sentence,
            copy_state=True,
        ):
            all_chunks.append(chunk.detach().cpu().numpy())

        if not all_chunks:
            return

        full_audio = np.concatenate(all_chunks)
        full_audio = np.clip(full_audio, -1, 1)
        full_audio = _trim_prefix(full_audio, native_rate)
        full_audio = _resample(full_audio, native_rate, target_rate)
        audio_int16 = _audio_to_int16(full_audio)

        # Yield WAV header + data
        yield _make_wav_header(target_rate, len(audio_int16))
        yield audio_int16.tobytes()

    def shutdown(self):
        """Release cached model and voice states to free memory."""
        global _model, _voice_states
        if _model is not None:
            LOG.info("Shutting down Pocket TTS — releasing model and voice states")
            _model = None
            _voice_states.clear()

    @staticmethod
    def available_languages() -> set:
        return {"en"}


PocketTTSPluginConfig = {
    "en": [
        {
            "voice": voice_name,
            "meta": {
                "gender": "male" if voice_name in ("marius", "javert", "jean") else "female",
                "display_name": f"Pocket TTS — {voice_name.title()}",
                "offline": True,
                "priority": 60,
            },
        }
        for voice_name in BUILTIN_VOICES
    ]
}
