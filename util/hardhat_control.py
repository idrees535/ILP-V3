import time 
import os 
import os
import pathlib
import subprocess
import logging
import brownie 

BASE_PATH = pathlib.Path().resolve().parent.as_posix()

HARDHAT_PROJECT_PATH = f"{BASE_PATH}/hardhat-project"   # Path to your Hardhat project

def start_hardhat_node():
    print("Starting Hardhat node...")
    try:
        # Start the Hardhat node as a background process
        process = subprocess.Popen(
            ["npx", "hardhat", "node"],
            cwd=os.path.join(BASE_PATH, "hardhat-project"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Hardhat node started successfully.")
    except FileNotFoundError:
        print("Error: 'npx' command not found. Ensure Node.js and npm are installed.")
    except Exception as e:
        print(f"An error occurred while starting Hardhat node: {e}")
    
    # Optionally, wait for a few seconds to ensure the node has started
    time.sleep(10)

# Function to stop the Hardhat node
def stop_hardhat_node():
    print("Stopping Hardhat node...")
    subprocess.run(["sudo", "fuser", "-k", "8545/tcp"])    # Kill any process using port 8545
    print("Hardhat node stopped.")

def brownie_connect():
    BROWNIE_PROJECTUniV3 = brownie.project.load(f"{BASE_PATH}/v3_core/", name="UniV3Project")
    #print(BROWNIE_PROJECTUniV3.available_contracts)

    if brownie.network.show_active() != "development":
        brownie.network.connect("development")
