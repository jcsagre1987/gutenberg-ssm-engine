@echo off
title Gutenberg SSM Edge Engine
color 0b
echo ===================================
echo    Launching Gutenberg Engine...
echo ===================================
cd /d "%~dp0"
start cmd /k "python server.py"
timeout /t 3 /nobreak > nul
start index.html