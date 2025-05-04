@echo off
cd /d %~dp0

REM Run the Java JAR file
start java -jar src\assets\XTABLES.jar

REM Change directory to src
cd src

REM Run the reef/object tracking agent
start python competitionDemo.py

REM Run the central/pathplanner agent
start python pathPlanner.py

REM Run the reef visualizer
start python runReefVisualizer.py

REM Prevent the window from closing immediately
pause
