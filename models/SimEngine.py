from agents.uniswap_lp_agent import UniswapV3LiquidityProviderAgent
from agents.uniswap_swapper_agent import UniswapV3SwapperAgent
from util.agent_policies import retail_lp_policy,noise_trader_policy


class SimEngine:
    def __init__(self, pool):
        self.pool = pool
        self.retail_lp = UniswapV3LiquidityProviderAgent(1e10,1e10,retail_lp_policy,self.pool)
        #self.inst_lp = UniswapV3LiquidityProviderAgent("inst_lp", 1000000000000.0,110000000000000.0,inst_LP_policy)
        #self.rl_lp = UniswapV3LiquidityProviderAgent("rl_lp", 1000000000000.0,110000000000000.0,rl_LP_policy)
        
        # Trader agents
        self.noise_trader = UniswapV3SwapperAgent(1e10,1e10, noise_trader_policy,self.pool)
        #self.whale_trader = UniswapV3SwapperAgent("whale_trader",5000000000000000.0,5500000000000000.0, whale_trader_policy)
    
    def run(self):
        self.retail_lp.takeStep()
        self.noise_trader.takeStep()
        # self.noise_trader.takeStep()