#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import subprocess
import time
from multiprocessing import Process

def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_app.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

def start_ganache():
    subprocess.run(["ganache-cli", "--quiet", "true", "--defaultBalanceEther", "10000000000000000000", "--gasLimit", "10000000000", "--gasPrice", "1", "--hardfork", "istanbul"])
    #print("Have you started ganache")

if __name__ == "__main__":
    # p1 = Process(target=start_ganache)
    # p1.start()

    # # sleep to wait util ganache started
    # time.sleep(5) 
    
    subprocess.run(["rm", "-rf", "../model_storage/liq_positions.json", "../model_storage/token_pool_addresses.json"])
    subprocess.run(["touch", "../model_storage/liq_positions.json", "../model_storage/token_pool_addresses.json"])
    p2 = Process(target=main)
    p2.start()
    
    # p1.join()
    p2.join()
