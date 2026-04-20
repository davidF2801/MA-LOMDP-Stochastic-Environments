@echo off
REM ================================================================================
REM Julia Launcher Script
REM ================================================================================
REM This script launches Julia with proper project environment setup
REM and handles Windows-specific path issues
REM ================================================================================

echo.
echo ================================================================================
echo Julia Launcher
echo ================================================================================
echo.

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo Current directory: %CD%
echo.

REM Check if Julia is available
where julia >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Julia is not found in PATH!
    echo Please ensure Julia is installed and added to your system PATH.
    echo.
    pause
    exit /b 1
)

echo Julia found. Launching with project environment...
echo.

REM Launch Julia with project activation
REM The --project=. flag ensures Julia uses the local Project.toml
julia --project=. --banner=no

echo.
echo Julia session ended.
pause


