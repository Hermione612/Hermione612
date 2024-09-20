# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
a = Analysis(
    ['main.py'],
    pathex=['C:\\Users\\lenovo\\try_software2'],
    binaries=[],
    datas=[('model_26.pth', '.'),
           ('model_45.pth', '.'),
           ('scaler_x_26.joblib', '.'),
           ('scaler_x_45.joblib', '.'),
           ('scaler_y_26.joblib', '.'),
           ('scaler_y_45.joblib', '.')],
    hiddenimports=['torch', 'torchvision', 'tensorboard'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='Prediction',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )