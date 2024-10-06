#!/bin/bash
sed -i 's/\r$//' brownie-compile.sh

set -e  # Stop script on any error

echo "compile.sh: Compile v3_core/..."
cd v3_core || { echo "Failed to change directory to v3_core"; exit 1; }

echo """compiler:
   solc:
       version: 0.7.6""" > brownie-config.yaml

# Run the Python script to install or compile solc
python3 ../solcx_install.py || { echo "Failed to run solcx_install.py"; exit 1; }

# Continue with the Brownie compilation
brownie compile || { echo "Brownie compilation failed"; exit 1; }

echo "compile.sh: Done"
