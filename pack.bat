echo delete cache
del /f /s /q dist\*
echo start make
pyinstaller gr.py -F
echo end make