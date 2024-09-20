# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
a = Analysis(
    ['main.py',"C:\\Users\\lenovo\\try_software2\\gcn_model.py"],
    pathex=['C:\\Users\\lenovo\\try_software2'],
    binaries=[],
    datas=[('C:\\Users\\lenovo\\try_software2\\model_beam.pth', '.weights'),
           ('C:\\Users\\lenovo\\try_software2\\model_platetri.pth', '.weights'),
           ('C:\\Users\\lenovo\\try_software2\\model_plateyi.pth', '.weights'),
           ('C:\\Users\\lenovo\\try_software2\\scaler_beam_y.joblib', '.parameters'),
           ('C:\\Users\\lenovo\\try_software2\\scaler_beam_x.joblib', '.parameters'),
           ('C:\\Users\\lenovo\\try_software2\\scaler_platetri_y.joblib', '.parameters'),
           ('C:\\Users\\lenovo\\try_software2\\scaler_platetri_x.joblib', '.parameters'),
           ('C:\\Users\\lenovo\\try_software2\\scaler_plateyi_y.joblib', '.parameters'),
           ('C:\\Users\\lenovo\\try_software2\\scaler_plateyi_x.joblib', '.parameters'),
           ('C:\\Users\\lenovo\\try_software2\\beam_distance.csv', '.distances'),
           ('C:\\Users\\lenovo\\try_software2\\plate_yi_distance1.csv', '.distances'),
           ('C:\\Users\\lenovo\\try_software2\\plate_tri_distance1.csv', '.distances')],
    hiddenimports=['torch','sklearn','openpyxl'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='C:\\Users\\lenovo\\try_software2\\cover.ico'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
