#!/bin/bash


cd src
# Run the reef/object tracking agent
python3 competitionDemo.py &

# Run the central/pathplanner agent
python3 pathPlanner.py &

# Run the reef visualizer
python3 runReefVisualizer.py &

# Wait for all background processes to finish
wait
