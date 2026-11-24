
# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

#HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

#EMBEDDING_MODEL = (
#    HF_CACHE
#    / "models--Qwen--Qwen3-Embedding-0.6B"
#    / "snapshots"
#    / "744169034862c8eec56628663995004342e4e449"
#)

#GENERATIVE_MODEL = (
#    HF_CACHE
#    / "models--Qwen--Qwen3-0.6B"
#    / "snapshots"
#    / "e6de91484c29aa9480d55605af694f39b081c455"
#)

#if not EMBEDDING_MODEL.is_dir():
#    raise FileNotFoundError(EMBEDDING_MODEL)

#if not GENERATIVE_MODEL.is_dir():
#    raise FileNotFoundError(GENERATIVE_MODEL)


a = Analysis(
    ["pdf_search_gui.py"],
    pathex=[],
    binaries=[],
	datas=[],
#    datas=[
#        (str(EMBEDDING_MODEL), "models/Qwen3-Embedding-0.6B"),
#        (str(GENERATIVE_MODEL), "models/Qwen3-0.6B"),
#    ],
    hiddenimports=[
        "transformers.models.qwen3",
        "faiss",
        "sentencepiece",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
	excludes=[
        "torchaudio",
        "torchvision",
        "tensorflow",
        "keras",
        "flax",
        "jax",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="RecherchePDF",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
