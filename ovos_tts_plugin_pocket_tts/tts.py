"""Pocket TTS plugin for OVOS — lightweight, CPU-only TTS by Kyutai."""

import os

import numpy as np
from ovos_plugin_manager.templates.tts import TTS
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


class PocketTTSPlugin(TTS):
    """Pocket TTS — 100M param, CPU-only, real-time TTS by Kyutai.

    Config example (mycroft.conf):
        "tts": {
            "module": "ovos-tts-plugin-pocket-tts",
            "ovos-tts-plugin-pocket-tts": {
                "voice": "alba"
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
        import scipy.io.wavfile

        voice = voice or self.config.get("voice", DEFAULT_VOICE)

        # Resolve voice — if it looks like a path or URI, use as-is;
        # otherwise treat as a built-in voice name
        if not (os.path.sep in voice or voice.startswith("hf://") or voice.endswith((".wav", ".safetensors"))):
            voice = voice.lower()

        model = _get_model()
        voice_state = _get_voice_state(model, voice)

        audio = model.generate_audio(voice_state, _PREFIX + sentence)
        audio_np = audio.numpy()
        audio_np = _trim_prefix(audio_np, model.sample_rate)

        scipy.io.wavfile.write(wav_file, model.sample_rate, audio_np)

        return wav_file, None

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
