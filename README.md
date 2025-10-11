# NEURA Glove
Project Repo for NEURA Glove

<!-- This sets up to run the tests and place a badge on GitHub if it passes -->


![Tests](https://github.com/SSOE-ECE1390/ExampleTeam/actions/workflows/tests.yml/badge.svg)


## NEURA Glove: Neural Enhanced User Reality Assistant

Current VR/AR systems rely on handheld controllers or optical tracking, which limit natural interaction. Controllers constrain motion, while camera-based tracking suffers from occlusion, lighting sensitivity, restricted range, and high computational costs.

The NEURA Glove addresses these issues using a wearable system with flex sensors (for finger bends) and an IMU (for hand orientation/motion). Sensor data is processed by a neural network to deliver accurate, real-time hand pose estimation—free from lighting and line-of-sight constraints.

Prototype Capabilities:
- High-accuracy finger pose recognition
- Smooth, low-jitter temporal tracking
- Seamless VR integration

Applications:
- Gaming & immersive interaction
- Rehabilitation therapy with precise motion tracking
- Accessibility via custom gesture-based input

By combining wearable sensing, machine learning, and VR integration, the NEURA Glove provides a cost-effective, portable, and inclusive alternative to traditional input systems.

Team Members:
Iyan Nekib (iyn1@pitt.edu)
Caitlyn Homa (cmh299@pitt.edu)
Lucas Connell (lzc6@pitt.edu)



## File Descriptions
This project contains a number of additional files that are used by GitHub to provide information and do tests on code.

### Markup files (*.md)
Markup files, such as this README file are shown on the home page of GitHub

[Here is a good reference for how to use markup files](https://github.com/lifeparticle/Markdown-Cheatsheet)

* README.md; This file usually holds information about the purpose of the repo, the authors, etc.  

* CODE_OF_CONDUCT.md; This file establishes a set of behavioral expectations for contributors and community members, promoting a positive and inclusive environment.

* LICENSE.md; This file specifies the licensing terms under which your project is released, informing users about how they can use, modify, and distribute your code.

### .gitignore
The .gitignore file is used to specify any files that should not be included in git commits/pushes.  Generally, these are temporary files or specific to your computer.  In this case, I have all the python environment files in the .venc folder flagged to be ignored.

### requirements.txt
The requirements.txt file is a way to specify the libraries needed by python by your code.  Here I have a general use one "requirements.txt" and one specifically used in the code regression testing "requirements_dev.txt".  Once you have your python install setup and running the way you like it, you can automatically generate the requirements.txt file for others to replicate your setup using the command

```
    pip freeze > requirements.txt
```

To install from a requirements.txt file use
```
    pip install -r requirements.txt
```
