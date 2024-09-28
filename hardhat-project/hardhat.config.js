/** @type import('hardhat/config').HardhatUserConfig */

module.exports = {
  solidity: "0.8.18",
  networks: {
    hardhat: {
      chainId: 1337, // Default chain ID for local Hardhat
      port: 8545,    // Specify the port
      gas: 2100000,  // Optional: You can set a gas limit (optional)
      gasPrice: 1000000000, // 20 gwei in wei
    },
  },
};


