from distutils.core import setup
import py2exe
import sys

# 如果运行时没有参数，添加一个用于 py2exe
if len(sys.argv) == 1:
    sys.argv.append('py2exe')

setup(
    options={
        'py2exe': {
            'packages': ['torch'],  # 确保包含 torch 和其他可能需要显式包含的库
            'includes': ['gcn_model'],  # 包含 gcn_model 模块
            'dll_excludes': ['MSVCP90.dll'],  # 有时可能需要排除特定的 DLL
            'dist_dir': 'dist',  # 指定输出目录
            'compressed': True,  # 压缩生成的文件
            'optimize': 2
        }
    },
    windows=[{
        'script': 'main.py',  # 主 Python 文件
        'icon_resources': [(1, 'cover.ico')] , # 程序图标文件
        'name': 'S2D'
    }],
    data_files=[
        ('', ['model_26.pth', 'model_45.pth', 'scaler_x_26.joblib', 'scaler_x_45.joblib', 'scaler_y_26.joblib', 'scaler_y_45.joblib']),  # 包含模型和标准化文件
    ],
    zipfile=None  # 所有库都打包进单个 exe 文件，或者指定一个库 zip 文件
)
