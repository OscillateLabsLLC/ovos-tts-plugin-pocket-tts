"""Pocket TTS plugin for OVOS — lightweight, CPU-only TTS by Kyutai."""

import inspect
import os
import struct
from typing import AsyncIterable, Dict, Optional

import numpy as np
from ovos_plugin_manager.templates.tts import TTS
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.log import LOG


# Lazy-loaded model cache, keyed by kyutai language id (e.g. "english_v2",
# "french_24l"). Single-language pocket-tts builds always live under the
# "english_v2" key so the rest of the code stays uniform.
_models: Dict[str, "object"] = {}
_voice_states: Dict[str, "object"] = {}

BUILTIN_VOICES = (
    "alba", "marius", "javert", "jean",
    "fantine", "cosette", "eponine", "azelma",
)

DEFAULT_VOICE = "alba"

# Map BCP-47 base language codes to the kyutai pocket-tts language id.
# Users can override or extend this via the `language_aliases` config key.
# Keys are the lower-cased base language (the part before the dash in a
# BCP-47 tag), values are the pocket-tts `language=` argument.
_DEFAULT_LANG_MAP: Dict[str, str] = {
    "en": "english_v2",
    "fr": "french_24l",
    "de": "german_24l",
    "es": "spanish_24l",
    "it": "italian_24l",
    "pt": "portuguese_24l",
}

# Cached at module import — does the installed pocket-tts expose the
# `language=` kwarg on `TTSModel.load_model`? See kyutai-labs/pocket-tts#155.
_MULTILINGUAL_SUPPORTED: Optional[bool] = None
_QUANTIZE_SUPPORTED: Optional[bool] = None


def _detect_pocket_tts_capabilities():
    """Inspect the installed pocket-tts to see which kwargs `load_model` accepts.

    Returns a (multilingual, quantize) tuple of booleans. Defers the import
    so plugin import stays cheap, but caches the result on first call.
    """
    global _MULTILINGUAL_SUPPORTED, _QUANTIZE_SUPPORTED
    if _MULTILINGUAL_SUPPORTED is not None:
        return _MULTILINGUAL_SUPPORTED, _QUANTIZE_SUPPORTED
    try:
        from pocket_tts import TTSModel
        params = inspect.signature(TTSModel.load_model).parameters
        _MULTILINGUAL_SUPPORTED = "language" in params
        _QUANTIZE_SUPPORTED = "quantize" in params
    except Exception as err:
        LOG.warning("Could not introspect pocket_tts.TTSModel.load_model: %s", err)
        _MULTILINGUAL_SUPPORTED = False
        _QUANTIZE_SUPPORTED = False
    return _MULTILINGUAL_SUPPORTED, _QUANTIZE_SUPPORTED

# Sacrificial prefix to prevent first-word swallowing.
# Pocket-tts blends the voice prompt into the start of generation, which can
# eat the first word. Prepending "..." produces a short throwaway utterance
# followed by a silence gap that we detect and trim.
_PREFIX = "... "
_PREFIX_MIN_DURATION = 0.15  # seconds — don't look for gap before this
_PREFIX_MAX_DURATION = 1.0   # seconds — stop looking after this
_PREFIX_SILENCE_GAP = 0.08   # seconds — minimum silence to count as gap
_SILENCE_THRESHOLD = 0.01    # fraction of max amplitude


def _resolve_lang(lang: Optional[str], lang_map: Dict[str, str]) -> str:
    """Map a BCP-47 language tag to a kyutai pocket-tts language id.

    Lookup order:
      1. Full BCP-47 tag (e.g. "pt-br") — lets users distinguish region
         variants when upstream eventually ships them.
      2. Base language subtag (e.g. "pt") — the common case.
      3. English fallback.

    On a legacy single-language pocket-tts build the result is always
    "english_v2"; the lookup is short-circuited.
    """
    multilingual, _ = _detect_pocket_tts_capabilities()
    if not multilingual:
        return "english_v2"
    if not lang:
        return lang_map.get("en", "english_v2")
    canonical = standardize_lang_tag(lang).lower()
    if canonical in lang_map:
        return lang_map[canonical]
    base = canonical.split("-")[0]
    return lang_map.get(base, lang_map.get("en", "english_v2"))


def _get_model(kyutai_lang: str, quantize: bool = False):
    """Lazy-load and cache the Pocket TTS model for a given kyutai language id."""
    if kyutai_lang in _models:
        return _models[kyutai_lang]
    from pocket_tts import TTSModel

    multilingual, quantize_supported = _detect_pocket_tts_capabilities()
    kwargs = {}
    if multilingual:
        kwargs["language"] = kyutai_lang
    if quantize_supported and quantize:
        kwargs["quantize"] = True

    LOG.info(
        "Loading Pocket TTS model (lang=%s, quantize=%s) — first call may download weights",
        kyutai_lang if multilingual else "english (legacy)",
        bool(kwargs.get("quantize", False)),
    )
    _models[kyutai_lang] = TTSModel.load_model(**kwargs)
    LOG.info("Pocket TTS model loaded for %s", kyutai_lang)
    return _models[kyutai_lang]


def _get_voice_state(model, voice: str, kyutai_lang: str):
    """Get or cache a voice state for the given voice identifier.

    voice can be:
      - A built-in voice name (e.g. "alba")
      - A path to a .wav file (voice cloning)
      - A path to a .safetensors file (exported voice state)
      - A HuggingFace URI (hf://kyutai/tts-voices/...)

    The cache is keyed by (kyutai_lang, voice) because voice state tensors
    are produced by a specific language model and are not interchangeable.
    """
    key = f"{kyutai_lang}::{voice}"
    if key not in _voice_states:
        LOG.info("Loading voice state for: %s (lang=%s)", voice, kyutai_lang)
        _voice_states[key] = model.get_state_for_audio_prompt(voice)
    return _voice_states[key]


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


class PocketTTSPlugin(TTS):
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
        # Preload requested languages so the first speak() doesn't pay the
        # download/load cost. Optional — most users will leave this empty
        # and let _get_model lazy-load on demand.
        for code in self.config.get("preload_languages", []):
            try:
                self._load_language(code)
            except Exception as err:
                LOG.warning("Failed to preload pocket-tts language %s: %s", code, err)

    @property
    def lang_map(self) -> Dict[str, str]:
        """Effective BCP-47 → kyutai language id map (default + user overrides)."""
        return {**_DEFAULT_LANG_MAP, **self.config.get("language_aliases", {})}

    def _quantize_for(self, kyutai_lang: str) -> bool:
        """Whether to enable quantization for a given kyutai language id.

        Defaults to True for the 24-layer preview models (where the maintainer
        recommends quantization for speed) and False for the distilled english
        defaults. Users can override per-language via the `quantize` config
        knob (bool or dict-of-booleans-keyed-by-kyutai-lang).
        """
        override = self.config.get("quantize")
        if isinstance(override, dict):
            if kyutai_lang in override:
                return bool(override[kyutai_lang])
        elif override is not None:
            return bool(override)
        return kyutai_lang.endswith("_24l")

    def _load_language(self, lang_or_code: str):
        """Resolve a BCP-47 (or kyutai) code and warm the model cache."""
        if lang_or_code in self.lang_map.values():
            kyutai_lang = lang_or_code
        else:
            kyutai_lang = _resolve_lang(lang_or_code, self.lang_map)
        return _get_model(kyutai_lang, quantize=self._quantize_for(kyutai_lang))

    def get_tts(self, sentence: str, wav_file: str, lang: str = None, voice: str = None) -> tuple:
        """Synchronous TTS with prefix trimming and resampling."""
        import wave

        voice = _resolve_voice(voice or self.config.get("voice", DEFAULT_VOICE))
        kyutai_lang = _resolve_lang(lang or self.lang, self.lang_map)
        model = _get_model(kyutai_lang, quantize=self._quantize_for(kyutai_lang))
        voice_state = _get_voice_state(model, voice, kyutai_lang)

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

    async def stream_tts(
        self,
        sentence: str,
        lang: Optional[str] = None,
        voice: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterable[bytes]:
        """Stream TTS audio chunks as they become available.

        Buffers the first ~1s of audio to trim the sacrificial prefix,
        then yields remaining chunks as raw PCM bytes wrapped in a WAV
        container.
        """
        voice = _resolve_voice(voice or self.config.get("voice", DEFAULT_VOICE))
        kyutai_lang = _resolve_lang(lang or self.lang, self.lang_map)
        model = _get_model(kyutai_lang, quantize=self._quantize_for(kyutai_lang))
        voice_state = _get_voice_state(model, voice, kyutai_lang)
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
        """Release cached models and voice states to free memory."""
        if _models:
            LOG.info("Shutting down Pocket TTS — releasing %d model(s) and voice states", len(_models))
            _models.clear()
            _voice_states.clear()

    @classmethod
    def available_languages(cls) -> set:
        """Return BCP-47 base language codes this plugin can serve.

        On a legacy single-language pocket-tts install this is just {"en"}.
        On a multilingual install (kyutai-labs/pocket-tts#155 or later) it
        returns the full set of mapped languages.
        """
        multilingual, _ = _detect_pocket_tts_capabilities()
        if not multilingual:
            return {"en"}
        return set(_DEFAULT_LANG_MAP.keys())


def _build_plugin_config():
    """Build the OPM PocketTTSPluginConfig advertisement.

    Emits one entry per (language, builtin_voice) pair so OPM can show
    every voice in the picker. Advertises the full multilingual set
    unconditionally — calling `_detect_pocket_tts_capabilities()` here
    would import `pocket_tts` at plugin-import time and blow the
    opm-check 500 ms budget. If a single-language pocket-tts is
    installed, the runtime resolver in `_resolve_lang` will fall back
    to english at speak() time.
    """
    config = {}
    for lang in sorted(_DEFAULT_LANG_MAP.keys()):
        config[lang] = [
            {
                "voice": voice_name,
                "lang": lang,
                "meta": {
                    "gender": "male" if voice_name in ("marius", "javert", "jean") else "female",
                    "display_name": f"Pocket TTS — {voice_name.title()} ({lang})",
                    "offline": True,
                    "priority": 60,
                },
            }
            for voice_name in BUILTIN_VOICES
        ]
    return config


PocketTTSPluginConfig = _build_plugin_config()
