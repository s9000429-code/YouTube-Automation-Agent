@echo off
rem Usage: yta.bat video make "topic"   or   yta.bat shorts create video.mp4
where py >/dev/null 2>/dev/null && (py -3 "%~dp0agent.py" %* & goto :eof)
python "%~dp0agent.py" %*
