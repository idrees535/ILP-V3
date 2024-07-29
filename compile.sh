#!/bin/bash

echo "compile.sh: Compile v3_core/..."
cd v3_core

echo """compiler:
   solc:
       version: 0.7.6""" > brownie-config.yaml

# Run the Python script to install or compile solc
python3 ../solcx_install.py

# Continue with the Brownie compilation
brownie compile

echo "compile.sh: Done"
