# Block Builder

A robotic arm system that autonomously perceives, picks, and assembles LEGO-style blocks into a target structure — built for Caltech's **ME134: Autonomous Robotic Manipulation** course.

## Overview

The robot uses a 5-DOF arm (HEBI actuators) with an Intel RealSense depth camera and ArUco markers to:

- Perceive a workspace grid and locate scattered blocks
- Plan an assembly (including an AI assembly-design agent built with Google's Agent Development Kit)
- Pick, verify, and place blocks to build the target structure, with recovery behavior for failed grips and periodic recalibration

The project progressed through ten milestones ("Goals 1–10") over the course of the term, from basic kinematics and object detection through full closed-loop assembly with placement verification, block-separation logic, and an end-of-class live demo.

## Demo

![Demo of the robot building a structure](media/demo.gif)

📄 [Full technical report](report.pdf) · 🎥 [Video](https://drive.google.com/file/d/1Z5a8N6V8IHZS3Czf7M2rwoayD5a_sxi2/view?usp=sharing) · 🖥️ [Presentation](https://docs.google.com/presentation/d/1z6xrhs0PZiR2KbXnW-9BohGcE1yS3LkcOwbhKHGI4iA/edit?usp=sharing)

## Stack

![ROS 2](https://img.shields.io/badge/ROS_2-22314E?style=flat-square&logo=ros&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)
![HEBI Robotics](https://img.shields.io/badge/HEBI_Robotics-333333?style=flat-square)

| Area | Techniques |
|---|---|
| **Perception** | HSV + contour block detection with shape classification · HDBSCAN/DBSCAN clustering for temporally stable detections · sparse voxel-grid world map with log-odds occupancy updates (RealSense depth) · grip-quality checks via monocular depth (Depth Anything V2) |
| **Planning** | 3-phase build state machine (grasp → approach → place) · center-outward, bottom-up placement ordering · reachability & block-separation logic for connected piles · Gemini-based ADK agent for natural-language structure design & validation |
| **Control** | Newton-Raphson IK on a Cartesian+tilt Jacobian with damped pseudoinverse · multi-body gravity compensation · 100 Hz quintic-spline trajectory interpolation · ArUco-based XY/Z calibration |
| **Building** | Post-placement verification & grip-failure recovery · live FastAPI dashboard for build monitoring |

## Layout

### ROS 2 node architecture

```
Nodes (4 executables, wired by src/legobuilder/launch/*.launch.py)
────────────────────────────────────────────────────────────────────
BrainNode(rclpy.Node)                         Central orchestrator. Owns the build state
  └── BuildPipeline                           machine and bridges perception, IK solving,
      ├── BlockStructure                      and motor control over ROS topics.
      ├── ObjectProcessor
      ├── StructureScanner
      ├── NextTargetBlock
      ├── Reachability
      ├── Separation
      ├── Recovery
      ├── Verification
      ├── UnreachableTracker
      ├── Precompute
      ├── Calibration
      └── Gestures
  (+ DashboardServer/Client)

DetectorNode(rclpy.Node)                      Perception. Runs the overhead-camera
  ├── Detector                                detection pipeline and builds a 3D
  ├── WorldMap                                occupancy map from the wrist depth camera.
  │   └── VoxelStore
  ├── Fusion
  └── GripAnalyzer

ManipulatorNode(rclpy.Node)                   Hard real-time loop at 100 Hz. Executes
  ├── Trajectory                              trajectories and talks directly to the
  ├── Gravity                                 HEBI motor hardware.
  └── BlockManipulator

IKSolverNode(rclpy.Node)                      Runs IK on a background thread in its own
  ├── BlockManipulator                        process, so heavy solves never block the
  └── KinematicChain                          Brain's perception callbacks.

Message flow
────────────────────────────────────────────────────────────────────
  Detector     → detector/contour_info, /world_map, /grid_coords →  Brain
  Brain        → brain/ik_request →                                 IKSolver
  IKSolver     → ik_solver/ik_response →                            Brain
  Brain        → brain/trajectory_command →                         Manipulator
  Manipulator  → manipulator/contact, manipulator/grip_failure →    Brain

  All four nodes also publish a periodic heartbeat topic so the
  others can detect when one has gone offline.
```

## Origin

This was a team project for ME134 (Winter 2026), originally developed on a shared lab machine. This repo is a snapshot of the final state of the codebase for portfolio purposes.
