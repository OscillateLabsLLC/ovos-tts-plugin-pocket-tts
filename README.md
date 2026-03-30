# ovos-tts-plugin-pocket-tts

[![Status: Proof of Concept](https://img.shields.io/badge/status-proof%20of%20concept-orange)](https://github.com/OscillateLabsLLC/.github/blob/main/SUPPORT_STATUS.md)

OVOS TTS plugin for [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) by Kyutai — a lightweight, CPU-only text-to-speech engine with a 100M parameter model.

~6x real-time on Apple Silicon, no GPU required. Supports voice cloning from audio samples.

## Install

```bash
pip install ovos-tts-plugin-pocket-tts
```

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

English only (for now — Kyutai's roadmap includes more languages).

## License

Apache-2.0
