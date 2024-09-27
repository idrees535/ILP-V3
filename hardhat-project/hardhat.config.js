/** @type import('hardhat/config').HardhatUserConfig */
// module.exports = {
//   solidity: "0.8.27",
// };
// module.exports = {
//   solidity: "0.8.18",
//   networks: {
//     hardhat: {
//       forking: {
//         url: "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID", // Optional for mainnet forking
//       },
//       chainId: 1337, // Default chain ID for local Hardhat
//       port: 8545,    // Specify the port
//     },
//   },
// };

module.exports = {
  solidity: "0.8.18",
  networks: {
    hardhat: {
      chainId: 1337, // Default chain ID for local Hardhat
      port: 8545,    // Specify the port
    },
  },
};

