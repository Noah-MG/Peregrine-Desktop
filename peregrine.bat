@echo off
setlocal EnableExtensions
REM ---------------------------------------------------------------------
REM Peregrine desktop wizard launcher.
REM
REM Double-click it, or call "peregrine" from any directory once this
REM folder is on your PATH. Any arguments are passed straight through.
REM ---------------------------------------------------------------------

set "HERE=%~dp0"
set "WIZ=%HERE%wizard\peregrine.py"

if not exist "%WIZ%" (
    echo.
    echo   Cannot find the wizard at:
    echo     "%WIZ%"
    echo   Keep peregrine.bat in the root of the Peregrine-Desktop folder.
    goto :fail
)

REM Find a Python 3.10+ interpreter. The Microsoft Store stub answers to
REM "python" but fails on launch, so each candidate is actually executed
REM rather than just located.
set "PYEXE="
call :trypy "py -3.12"
if not defined PYEXE call :trypy "py -3"
if not defined PYEXE call :trypy "python3"
if not defined PYEXE call :trypy "python"

if not defined PYEXE (
    echo.
    echo   No usable Python 3.10+ found.
    echo   Install Python from https://www.python.org/downloads/ and make
    echo   sure the "py" launcher is included, then run this again.
    goto :fail
)

%PYEXE% "%WIZ%" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo   Wizard exited with code %RC%.
)
call :maybepause
endlocal & exit /b %RC%


:trypy
REM Run the candidate and check the version actually works.
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "PYEXE=%~1"
goto :eof


:maybepause
REM Keep the window open only when launched by double-click. When run from
REM an existing console, %cmdcmdline% does not mention this script's name.
echo %cmdcmdline% | find /i "%~nx0" >nul
if not errorlevel 1 (
    echo.
    pause
)
goto :eof


:fail
call :maybepause
endlocal & exit /b 1
