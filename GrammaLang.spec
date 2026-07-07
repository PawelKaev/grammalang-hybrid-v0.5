# -*- mode: python ; coding: utf-8 -*-

import os, glob

block_cipher = None

# DLL от llama-cpp-python
llama_dir = os.path.dirname(__import__('llama_cpp').__file__)
llama_lib = os.path.join(llama_dir, 'lib')
llama_dlls = glob.glob(os.path.join(llama_lib, '*.dll'))
binaries = [(dll, 'llama_cpp/lib') for dll in llama_dlls]

a = Analysis(
    ['gui_app.py'],
    pathex=['C:\\Projects\\local-ai-one-button', 'C:\\Users\\Валерий\\Desktop\\Grammalang\\hybrid'],
    binaries=binaries,
    datas=[
        ('prompts/', 'prompts/'),
        ('configs/', 'configs/'),
        ('src/', 'src/'),
    ],
    hiddenimports=[
        'llama_cpp',
        'llama_cpp.llama_cpp',
        'src.fast_parser',
        'src.deep_interpreter',
        'src.fusion',
        'local_model',
        'pydantic',
        'json',
        'requests',
        're',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='GrammaLang',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
