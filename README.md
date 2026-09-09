# Block Builder

A robotic arm system that autonomously perceives, picks, and assembles LEGO-style blocks into a target structure — built for Caltech's **ME134: Autonomous Robotic Manipulation** course.

## Overview

The robot uses a 5-DOF arm (HEBI actuators) with an Intel RealSense depth camera and ArUco markers to:

- Perceive a workspace grid and locate scattered blocks
- Plan an assembly (including an AI assembly-design agent built with Google's Agent Development Kit)
- Pick, verify, and place blocks to build the target structure, with recovery behavior for failed grips and periodic recalibration

The project progressed through ten milestones ("Goals 1–10") over the course of the term, from basic kinematics and object detection through full closed-loop assembly with placement verification, block-separation logic, and an end-of-class live demo.

## Demo

<video src="https://github.com/SebastianBa28/block-builder/raw/main/media/demo.mp4" controls poster="media/poster.jpg" width="720">
  Your browser doesn't support embedded video — <a href="media/demo.mp4">watch the demo here</a>.
</video>

## Stack

ROS 2 · Python · HEBI Robotics SDK · OpenCV · PyTorch/Transformers (depth + perception) · Google ADK · FastAPI (live demo dashboard)

## Layout

- `src/legobuilder` — core ROS package: perception, kinematics/IK, build pipeline, block separation & placement verification
- `src/threedof` — arm URDF/meshes and low-level manipulator control
- `src/detectors`, `src/usb_cam`, `src/hebiros` — camera and actuator interface nodes
- `adk_agents/assembly_designer` — LLM agent that designs the block assembly from a goal description
- `scripts` — helper scripts to run the agent, the demo dashboard, and structure visualization
- `data` — recorded run logs (`.mcap`), calibration/grip test data, and result plots

## Origin

This was a team project for ME134 (Winter 2026), originally developed on a shared lab machine. This repo is a snapshot of the final state of the codebase for portfolio purposes.
