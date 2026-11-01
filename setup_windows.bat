@echo off
rem One-time setup. Double-click this file.
where py >/dev/null 2>/dev/null && (py -3 "%~dp0setup.py" %* & goto done)
where python >/dev/null 2>/dev/null && (python "%~dp0setup.py" %* & goto done)
echo Python 3 is not installed.
echo Install it from https://www.python.org/downloads/windows/ and tick "Add python.exe to PATH", then run this file again.
:done
pause
