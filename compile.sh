#Uniswapv3 contracts

echo "compile.sh: Compile v3_core/..."
cd v3_core
echo """compiler:
   solc:
       version: 0.7.6""" > brownie-config.yaml
brownie compile

echo ""

echo "compile.sh: Done"

