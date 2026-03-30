"""Basic tests for ovos-tts-plugin-pocket-tts."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_import_fast():
    """Plugin must import in < 500ms (no heavy deps at import time)."""
    import importlib
    import time

    start = time.monotonic()
    importlib.import_module("ovos_tts_plugin_pocket_tts")
    elapsed = time.monotonic() - start
    assert elapsed < 0.5, f"Import took {elapsed:.2f}s — must be < 0.5s for opm-check"


def test_available_languages():
    from ovos_tts_plugin_pocket_tts import PocketTTSPlugin

    langs = PocketTTSPlugin.available_languages()
    assert "en" in langs


def test_config_dict():
    from ovos_tts_plugin_pocket_tts import PocketTTSPluginConfig

    assert "en" in PocketTTSPluginConfig
    voices = PocketTTSPluginConfig["en"]
    assert len(voices) == 8
    names = {v["voice"] for v in voices}
    assert "alba" in names
    assert "marius" in names


@patch("ovos_tts_plugin_pocket_tts.tts._get_model")
def test_get_tts(mock_get_model, tmp_path):
    import wave

    # Create a fake audio tensor that supports .clamp().numpy()
    sample_rate = 24000
    t = np.linspace(0, 1.0, sample_rate, dtype=np.float32)
    fake_audio = np.sin(2 * np.pi * 440 * t)

    mock_model = MagicMock()
    mock_model.sample_rate = sample_rate
    mock_audio = MagicMock()
    mock_audio.clamp.return_value = mock_audio
    mock_audio.numpy.return_value = fake_audio
    mock_model.generate_audio.return_value = mock_audio
    mock_model.get_state_for_audio_prompt.return_value = "fake_state"
    mock_get_model.return_value = mock_model

    from ovos_tts_plugin_pocket_tts import PocketTTSPlugin

    plug = PocketTTSPlugin(config={"lang": "en", "voice": "alba"})
    wav_file = str(tmp_path / "test.wav")
    result, phonemes = plug.get_tts("Hello world", wav_file)

    assert result == wav_file
    assert phonemes is None
    mock_model.generate_audio.assert_called_once()
    mock_audio.clamp.assert_called_once_with(-1, 1)
    # Verify the prefix was prepended
    call_args = mock_model.generate_audio.call_args
    assert call_args[0][1].startswith("... ")
    # Verify valid 16-bit PCM WAV
    with wave.open(wav_file, "rb") as wf:
        assert wf.getsampwidth() == 2
        assert wf.getnchannels() == 1
        assert wf.getframerate() == sample_rate


def test_trim_prefix_with_silence_gap():
    """Verify prefix trimming detects a silence gap and removes the prefix."""
    from ovos_tts_plugin_pocket_tts.tts import _trim_prefix

    sr = 24000
    # Build: 0.2s of noise + 0.1s silence + 0.5s of signal
    prefix_noise = np.random.uniform(-0.5, 0.5, int(sr * 0.2)).astype(np.float32)
    silence = np.zeros(int(sr * 0.1), dtype=np.float32)
    signal = np.random.uniform(-0.5, 0.5, int(sr * 0.5)).astype(np.float32)

    audio = np.concatenate([prefix_noise, silence, signal])
    trimmed = _trim_prefix(audio, sr)

    # Should be significantly shorter than original (prefix removed)
    assert len(trimmed) < len(audio)
    # Should still contain most of the signal portion
    assert len(trimmed) >= int(sr * 0.4)


def test_trim_prefix_no_gap():
    """If no silence gap found, audio is returned mostly intact."""
    from ovos_tts_plugin_pocket_tts.tts import _trim_prefix

    sr = 24000
    # Continuous signal with no silence gap
    audio = np.random.uniform(-0.5, 0.5, sr).astype(np.float32)
    trimmed = _trim_prefix(audio, sr)

    # Should be roughly the same length (only leading/trailing silence trim)
    assert len(trimmed) > int(sr * 0.8)
