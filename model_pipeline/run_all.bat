@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "calib_images" (
    mkdir calib_images
    echo [ERROR] 请先在 model_pipeline\calib_images 中放入 30~50 张校准图片后再运行！
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [1/5] 正在创建虚拟环境 .venv ...
    python -m venv .venv || goto :err
)

call .venv\Scripts\activate.bat

echo [2/5] 正在安装依赖 ...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || goto :err

echo [3/5] 正在导出 FP32 ONNX ...
python export_onnx.py || goto :err

echo [4/5] 正在执行 INT8 PTQ 静态量化 ...
python quantize_ptq.py || goto :err

echo [5/5] 正在执行黄金样本精度对齐 ...
python verify_alignment.py || goto :err

echo.
echo [SUCCESS] 全流程跑通！产物已输出到 ..\artifacts\
pause
exit /b 0

:err
echo.
echo [FAILED] 执行出错，请查看上方报错信息。
pause
exit /b 1