# Changelog

## 0.1.0 (2026-04-09)


### Features

* add Dockerfile for ovos-tts-server with pocket-tts ([7ec94b6](https://github.com/OscillateLabsLLC/ovos-tts-plugin-pocket-tts/commit/7ec94b615e3d995ca317c130b5e4aab496c8f9cf))
* implement shutdown() to release model and voice state memory ([ca02450](https://github.com/OscillateLabsLLC/ovos-tts-plugin-pocket-tts/commit/ca024503f421ac35d185765edf6d1cfe8307b0dc))
* implement StreamingTTS with generate_audio_stream support ([e2080b5](https://github.com/OscillateLabsLLC/ovos-tts-plugin-pocket-tts/commit/e2080b555e0c69e957cbfde4600dafa5cb92edec))
* initial OVOS TTS plugin for Pocket TTS (Kyutai) ([f210697](https://github.com/OscillateLabsLLC/ovos-tts-plugin-pocket-tts/commit/f210697fb3ffa9db97244fef6b66558fb4ea6eb5))
* multilingual support with capability-based feature detection ([5c8c520](https://github.com/OscillateLabsLLC/ovos-tts-plugin-pocket-tts/commit/5c8c520264761e4d689d55b16cab997e327bd945))
* multilingual support with capability-based feature detection ([c6b1262](https://github.com/OscillateLabsLLC/ovos-tts-plugin-pocket-tts/commit/c6b12626ee4493343e29488c59e31d47085023fd))
* re-enable sacrificial prefix for improved first-word clarity ([a36a203](https://github.com/OscillateLabsLLC/ovos-tts-plugin-pocket-tts/commit/a36a20315439174dd20859f9ce2dc90a2ee092e6))


### Bug Fixes

* clamp audio to [-1,1] and write 16-bit PCM WAV ([7c75f42](https://github.com/OscillateLabsLLC/ovos-tts-plugin-pocket-tts/commit/7c75f42c319e33dcd44aeb1fc0d6a8d363858f9e))
* remove sacrificial prefix — likely causing scattered audio artifacts ([9bb8503](https://github.com/OscillateLabsLLC/ovos-tts-plugin-pocket-tts/commit/9bb85034e14a08204e55e4161ba6db742dba7a9e))
* resample to 16kHz (OVOS standard) for playback compatibility ([f52d8d8](https://github.com/OscillateLabsLLC/ovos-tts-plugin-pocket-tts/commit/f52d8d8d85c721ea22e616b154df336b19a345e9))
* revert base class to TTS for backwards compatibility ([3a1021f](https://github.com/OscillateLabsLLC/ovos-tts-plugin-pocket-tts/commit/3a1021fe9948bcc8400d19425f2df672111da1c1))


### Documentation

* add CPU-only torch guidance and uv source for lighter installs ([0c6768f](https://github.com/OscillateLabsLLC/ovos-tts-plugin-pocket-tts/commit/0c6768fcb26832f825ea9ff8cefb3c2faa84b3dd))
