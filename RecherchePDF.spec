
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ["pdf_search_gui.py"],
    pathex=[],
    binaries=[],
	datas=[],
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
