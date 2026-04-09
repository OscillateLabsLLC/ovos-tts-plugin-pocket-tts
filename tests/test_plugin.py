"""Basic tests for ovos-tts-plugin-pocket-tts."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_import_fast():
    """Plugin must import quickly (no heavy deps at import time).

    The opm-check budget is ~1s; we keep some headroom but accept that
    pulling in ovos-plugin-manager + ovos-utils alone is ~300ms on
    Python 3.13. Anything substantially over 1s means we accidentally
    imported pocket-tts or torch at module load.
    """
    import importlib
    import time

    start = time.monotonic()
    importlib.import_module("ovos_tts_plugin_pocket_tts")
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"Import took {elapsed:.2f}s — must be < 1.0s for opm-check"


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

    # Pin sample_rate to the model's native rate so we can assert
    # the WAV header without exercising the resample path.
    plug = PocketTTSPlugin(config={"lang": "en", "voice": "alba", "sample_rate": sample_rate})
    wav_file = str(tmp_path / "test.wav")
    result, phonemes = plug.get_tts("Hello world", wav_file)

    assert result == wav_file
    assert phonemes is None
    mock_model.generate_audio.assert_called_once()
    mock_audio.clamp.assert_called_once_with(-1, 1)
    # Verify the prefix was prepended
    call_args = mock_model.generate_audio.call_args
    assert call_args[0][1].startswith("... ")
    # Verify valid 16-bit PCM WAV at the native rate
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


# ---- Multilingual ------------------------------------------------------------


def _reset_capability_cache():
    """Clear the cached pocket-tts capability detection between tests."""
    from ovos_tts_plugin_pocket_tts import tts as tts_mod
    tts_mod._MULTILINGUAL_SUPPORTED = None
    tts_mod._QUANTIZE_SUPPORTED = None
    tts_mod._models.clear()
    tts_mod._voice_states.clear()


@pytest.fixture(autouse=True)
def _clear_caches():
    _reset_capability_cache()
    yield
    _reset_capability_cache()


def test_resolve_lang_multilingual():
    """When pocket-tts supports `language=`, BCP-47 tags map to kyutai ids."""
    from ovos_tts_plugin_pocket_tts import tts as tts_mod

    with patch.object(tts_mod, "_detect_pocket_tts_capabilities", return_value=(True, True)):
        m = tts_mod._DEFAULT_LANG_MAP
        assert tts_mod._resolve_lang("fr-FR", m) == "french_24l"
        assert tts_mod._resolve_lang("de-DE", m) == "german_24l"
        assert tts_mod._resolve_lang("es-ES", m) == "spanish_24l"
        assert tts_mod._resolve_lang("en-US", m) == "english_v2"
        # Unknown lang falls back to english.
        assert tts_mod._resolve_lang("ja-JP", m) == "english_v2"
        # None / empty falls back too.
        assert tts_mod._resolve_lang(None, m) == "english_v2"


def test_resolve_lang_legacy():
    """On a legacy single-language pocket-tts, every input collapses to english."""
    from ovos_tts_plugin_pocket_tts import tts as tts_mod

    with patch.object(tts_mod, "_detect_pocket_tts_capabilities", return_value=(False, False)):
        assert tts_mod._resolve_lang("fr-FR", tts_mod._DEFAULT_LANG_MAP) == "english_v2"
        assert tts_mod._resolve_lang("de-DE", tts_mod._DEFAULT_LANG_MAP) == "english_v2"


def test_resolve_lang_alias_override():
    """User-supplied language_aliases override the default map."""
    from ovos_tts_plugin_pocket_tts import tts as tts_mod

    custom = {**tts_mod._DEFAULT_LANG_MAP, "en": "english_v1", "fr": "french_24l"}
    with patch.object(tts_mod, "_detect_pocket_tts_capabilities", return_value=(True, True)):
        assert tts_mod._resolve_lang("en-US", custom) == "english_v1"
        assert tts_mod._resolve_lang("fr-FR", custom) == "french_24l"


def test_resolve_lang_full_tag_takes_precedence():
    """A full BCP-47 entry in the alias map wins over the base subtag.

    This is the future-proofing path for region variants like pt-BR vs
    pt-PT once upstream ships separate models.
    """
    from ovos_tts_plugin_pocket_tts import tts as tts_mod

    custom = {
        **tts_mod._DEFAULT_LANG_MAP,
        "pt-br": "portuguese_brazil_24l",  # hypothetical future model
    }
    with patch.object(tts_mod, "_detect_pocket_tts_capabilities", return_value=(True, True)):
        # Full-tag match wins.
        assert tts_mod._resolve_lang("pt-BR", custom) == "portuguese_brazil_24l"
        # Base-tag fallback still works for the unmapped variant.
        assert tts_mod._resolve_lang("pt-PT", custom) == "portuguese_24l"


def test_get_model_passes_language_kwarg():
    """When pocket-tts is multilingual, _get_model passes `language=...`."""
    from ovos_tts_plugin_pocket_tts import tts as tts_mod

    fake_model = MagicMock(name="fake_model")
    fake_TTSModel = MagicMock()
    fake_TTSModel.load_model.return_value = fake_model
    fake_pocket_tts = MagicMock(TTSModel=fake_TTSModel)

    with patch.dict("sys.modules", {"pocket_tts": fake_pocket_tts}), \
         patch.object(tts_mod, "_detect_pocket_tts_capabilities", return_value=(True, True)):
        result = tts_mod._get_model("french_24l", quantize=True)

    assert result is fake_model
    fake_TTSModel.load_model.assert_called_once_with(language="french_24l", quantize=True)


def test_get_model_legacy_no_language_kwarg():
    """On legacy pocket-tts, _get_model omits `language=` to stay compatible."""
    from ovos_tts_plugin_pocket_tts import tts as tts_mod

    fake_model = MagicMock(name="fake_model")
    fake_TTSModel = MagicMock()
    fake_TTSModel.load_model.return_value = fake_model
    fake_pocket_tts = MagicMock(TTSModel=fake_TTSModel)

    with patch.dict("sys.modules", {"pocket_tts": fake_pocket_tts}), \
         patch.object(tts_mod, "_detect_pocket_tts_capabilities", return_value=(False, False)):
        tts_mod._get_model("english_v2", quantize=False)

    fake_TTSModel.load_model.assert_called_once_with()


def test_available_languages_multilingual():
    from ovos_tts_plugin_pocket_tts import tts as tts_mod
    from ovos_tts_plugin_pocket_tts import PocketTTSPlugin

    with patch.object(tts_mod, "_detect_pocket_tts_capabilities", return_value=(True, True)):
        langs = PocketTTSPlugin.available_languages()
    assert {"en", "fr", "de", "es", "it", "pt"}.issubset(langs)


def test_available_languages_legacy():
    from ovos_tts_plugin_pocket_tts import tts as tts_mod
    from ovos_tts_plugin_pocket_tts import PocketTTSPlugin

    with patch.object(tts_mod, "_detect_pocket_tts_capabilities", return_value=(False, False)):
        langs = PocketTTSPlugin.available_languages()
    assert langs == {"en"}


def test_quantize_for_24l_default_on():
    """24-layer preview models default to quantize=True per maintainer recommendation."""
    from ovos_tts_plugin_pocket_tts import PocketTTSPlugin

    plug = PocketTTSPlugin(config={"lang": "en"})
    assert plug._quantize_for("french_24l") is True
    assert plug._quantize_for("english_v2") is False


def test_quantize_for_user_override():
    """User can force quantize on/off via config knob."""
    from ovos_tts_plugin_pocket_tts import PocketTTSPlugin

    plug_off = PocketTTSPlugin(config={"lang": "en", "quantize": False})
    assert plug_off._quantize_for("french_24l") is False

    plug_on = PocketTTSPlugin(config={"lang": "en", "quantize": True})
    assert plug_on._quantize_for("english_v2") is True


def test_quantize_for_per_lang_dict():
    """Quantize config can be a per-language dict."""
    from ovos_tts_plugin_pocket_tts import PocketTTSPlugin

    plug = PocketTTSPlugin(config={
        "lang": "en",
        "quantize": {"french_24l": False, "english_v2": True},
    })
    assert plug._quantize_for("french_24l") is False
    assert plug._quantize_for("english_v2") is True


@patch("ovos_tts_plugin_pocket_tts.tts._get_model")
def test_get_tts_threads_lang(mock_get_model, tmp_path):
    """get_tts() calls _get_model() with the resolved kyutai language id."""
    from ovos_tts_plugin_pocket_tts import tts as tts_mod
    from ovos_tts_plugin_pocket_tts import PocketTTSPlugin

    sample_rate = 24000
    fake = np.zeros(sample_rate, dtype=np.float32)
    mock_model = MagicMock()
    mock_model.sample_rate = sample_rate
    mock_audio = MagicMock()
    mock_audio.clamp.return_value = mock_audio
    mock_audio.numpy.return_value = fake
    mock_model.generate_audio.return_value = mock_audio
    mock_model.get_state_for_audio_prompt.return_value = "fake_state"
    mock_get_model.return_value = mock_model

    with patch.object(tts_mod, "_detect_pocket_tts_capabilities", return_value=(True, True)):
        plug = PocketTTSPlugin(config={"lang": "fr-FR", "voice": "alba"})
        wav = str(tmp_path / "fr.wav")
        plug.get_tts("Bonjour", wav, lang="fr-FR")

    # First positional arg is the kyutai language id
    args, kwargs = mock_get_model.call_args
    assert args[0] == "french_24l"
    assert kwargs.get("quantize") is True


@patch("ovos_tts_plugin_pocket_tts.tts._get_model")
def test_get_tts_uses_self_lang_when_unspecified(mock_get_model, tmp_path):
    """If no lang= is passed, get_tts uses self.lang from config."""
    from ovos_tts_plugin_pocket_tts import tts as tts_mod
    from ovos_tts_plugin_pocket_tts import PocketTTSPlugin

    sample_rate = 24000
    mock_model = MagicMock()
    mock_model.sample_rate = sample_rate
    mock_audio = MagicMock()
    mock_audio.clamp.return_value = mock_audio
    mock_audio.numpy.return_value = np.zeros(sample_rate, dtype=np.float32)
    mock_model.generate_audio.return_value = mock_audio
    mock_model.get_state_for_audio_prompt.return_value = "fake_state"
    mock_get_model.return_value = mock_model

    with patch.object(tts_mod, "_detect_pocket_tts_capabilities", return_value=(True, True)):
        plug = PocketTTSPlugin(config={"lang": "de-DE", "voice": "alba"})
        plug.get_tts("Hallo", str(tmp_path / "de.wav"))

    args, _ = mock_get_model.call_args
    assert args[0] == "german_24l"


def test_plugin_config_advertises_all_languages():
    from ovos_tts_plugin_pocket_tts import PocketTTSPluginConfig

    for lang in ("en", "fr", "de", "es", "it", "pt"):
        assert lang in PocketTTSPluginConfig
        voices = PocketTTSPluginConfig[lang]
        assert len(voices) == 8
        names = {v["voice"] for v in voices}
        assert "alba" in names
