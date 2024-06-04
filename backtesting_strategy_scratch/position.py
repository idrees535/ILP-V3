
import UNI_v3_funcs


class Position:

    def __init__(self,tick,tickLower,tickUpper,amount0,amount1,decimal0,decimal1) -> None:

        self.tick = tick
        self.tickLower = tickLower
        self.tickUpper = tickUpper

        self.decimal0 = decimal0
        self.decimal1 = decimal1

        self.positionLiquidity = UNI_v3_funcs.get_liquidity(tick,tickLower,tickUpper,amount0,amount1,decimal0,decimal1)
        self.depositedAmounts = UNI_v3_funcs.get_amounts(tick,tickLower,tickUpper,self.positionLiquidity,decimal0,decimal1)


        self.balance0 = amount0 - self.depositedAmounts[0]
        self.balance1 = amount1 - self.depositedAmounts[1]

        self.fee0 = 0
        self.fee1 = 0

        self.rebaseCount = 0

        pass

    def simulate():
        pass

    def deposit():
        pass

    def swap():
        pass

