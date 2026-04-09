# ovos-tts-plugin-pocket-tts

[![Status: Proof of Concept](https://img.shields.io/badge/status-proof%20of%20concept-orange)](https://github.com/OscillateLabsLLC/.github/blob/main/SUPPORT_STATUS.md)

OVOS TTS plugin for [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) by Kyutai — a lightweight, CPU-only text-to-speech engine with a 100M parameter model.

~6x real-time on Apple Silicon, no GPU required. Supports voice cloning from audio samples.

## Install

```bash
pip install ovos-tts-plugin-pocket-tts
```

With `uv`, torch automatically resolves to the CPU-only wheel (configured via `tool.uv.sources`).

### Linux: CPU-only torch (saves ~2GB)

On Linux, pip defaults to the CUDA torch wheel (~2.5GB). If you don't need GPU support (pocket-tts is CPU-only anyway), install torch from the CPU index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ovos-tts-plugin-pocket-tts
```

On macOS, this is not needed — PyPI torch is already CPU-only (~60MB).

## Configuration

```json
{
  "tts": {
    "module": "ovos-tts-plugin-pocket-tts",
    "ovos-tts-plugin-pocket-tts": {
      "voice": "alba"
    }
  }
}
```

### Voice options

**Built-in voices:** `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`

**Voice cloning:** Set `voice` to a path to a `.wav` file, a pre-exported `.safetensors` file, or a HuggingFace URI (`hf://kyutai/tts-voices/...`).

## Language support

This plugin auto-detects which languages your installed `pocket-tts` supports.

- **Legacy `pocket-tts` (≤ 1.1.1, current PyPI release):** English only.
- **Multilingual `pocket-tts` (kyutai-labs/pocket-tts#155, not yet released):** English, French, German, Spanish, Italian, Portuguese.

When the multilingual release lands on PyPI, just `pip install -U pocket-tts` and the plugin lights up the new languages automatically — no plugin upgrade required.

### How language selection works

The plugin maps the active OVOS language (BCP-47, e.g. `fr-FR`) to a kyutai pocket-tts model id. Defaults:

| OVOS lang | Kyutai model     |
| --------- | ---------------- |
| `en`      | `english_v2`     |
| `fr`      | `french_24l`     |
| `de`      | `german_24l`     |
| `es`      | `spanish_24l`    |
| `it`      | `italian_24l`    |
| `pt`      | `portuguese_24l` |

Lookup tries the full BCP-47 tag first (e.g. `pt-br`), then falls back to the base subtag (`pt`), then to English. Unknown languages fall back to English with a log line.

> **Note:** The `_24l` (24-layer) models are larger preview builds, not the final distilled releases. They are slower than the english defaults but support int8 quantization for a meaningful speedup. The plugin enables `quantize=True` automatically for any `*_24l` model. You can override per-language via the `quantize` config knob.

### Override the language map

If you want, for example, the older `english_v1` model, or to register a future regional model, set `language_aliases` in your config:

```json
{
  "tts": {
    "module": "ovos-tts-plugin-pocket-tts",
    "ovos-tts-plugin-pocket-tts": {
      "voice": "alba",
      "language_aliases": {
        "en": "english_v1",
        "pt-br": "portuguese_brazil_24l"
      },
      "quantize": {
        "french_24l": false
      },
      "preload_languages": ["en", "fr"]
    }
  }
}
```

| Key                  | Type           | Default                       | Description                                                                           |
| -------------------- | -------------- | ----------------------------- | ------------------------------------------------------------------------------------- |
| `voice`              | str            | `"alba"`                      | Built-in voice name, `.wav`/`.safetensors` path, or `hf://` URI.                      |
| `sample_rate`        | int            | `16000`                       | Output sample rate in Hz. The model is resampled if it differs.                       |
| `language_aliases`   | dict           | `{}`                          | Override or extend the BCP-47 → kyutai model map. Full tags take precedence over base subtags. |
| `quantize`           | bool or dict   | auto (`True` for `*_24l`)     | Force int8 quantization on/off. Pass a dict to control per kyutai model id.            |
| `preload_languages`  | list[str]      | `[]`                          | BCP-47 codes to load eagerly during plugin init instead of lazy-loading on first use.  |
| `enable_streaming`   | bool           | `false`                       | Use the streaming TTS path (recommended for low-latency setups).                       |

> **Memory note:** Each loaded language model holds ~100M parameters in RAM (the `*_24l` previews are ~4× bigger before distillation). The plugin caches one model per used language, so leaving `preload_languages` empty and letting the cache warm on demand keeps the resident set small.

## Roadmap

- **SouraTTS** (emotional TTS built on pocket-tts) is being tracked as a separate plugin (`ovos-tts-plugin-soura-tts`) — its emotion/intensity dimensions don't fit the current OVOS TTS interface and bundling would tie pocket-tts upgrades to SouraTTS releases.

## License

Apache-2.0
