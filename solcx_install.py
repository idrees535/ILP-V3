import solcx

# Set the installed version as the default compiler
solcx.set_solc_version('0.7.6')

# Check if the version is set correctly
print(solcx.get_solc_version())
