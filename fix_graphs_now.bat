@echo off
echo Fixing graph output issues (Unicode-safe version)...
echo.

python fix_graph_output_fixed.py

echo.
echo Graph fix completed. Check the output above for results.
echo.
echo If graphs still don't show, they will be saved as PNG files.
echo Press any key to exit...
pause >nul