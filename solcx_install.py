import solcx

# Install specific version of solc
solcx.install_solc('0.7.6')

# Set the installed version as the default compiler
solcx.set_solc_version('0.7.6')

# Check if the version is set correctly
print(f"Installed and set solc version: {solcx.get_solc_version()}")
