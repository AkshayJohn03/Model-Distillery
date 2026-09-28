"""Quantization layer: GGUF (llama.cpp) and AWQ (autoawq) command builders.

Both modules are configuration + command builders with README-grade docs. They
require local tooling (llama.cpp binaries / the autoawq package) and are **not
executed in CI**; the dry-run tests only assert that generated commands are
well-formed.
"""
