# solcx_install.py
import solcx

try:
    # Try to install the binary version first
    solcx.install_solc('0.7.6')
except solcx.exceptions.SolcInstallationError:
    print("Binary installation failed. Attempting to compile from source...")
    try:
        # Attempt to compile from source
        solcx.compile_solc('0.7.6')
    except Exception as e:
        print(f"Error compiling solc from source: {e}")
        exit(1)
