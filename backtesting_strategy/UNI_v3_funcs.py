#sqrtP: format X96 = int(1.0001**(tick/2)*(2**96))
#liquidity: int
#sqrtA = price for lower tick
#sqrtB = price for upper tick
'''get_amounts function'''
#Use 'get_amounts' function to calculate amounts as a function of liquitidy and price range
def get_amount0(sqrtA,sqrtB,liquidity,decimals):
    
    if (sqrtA > sqrtB):
          (sqrtA,sqrtB)=(sqrtB,sqrtA)
    
    amount0=((liquidity*2**96*(sqrtB-sqrtA)/sqrtB/sqrtA)/10**decimals)
    
    return amount0

def get_amount1(sqrtA,sqrtB,liquidity,decimals):
    
    if (sqrtA > sqrtB):
        (sqrtA,sqrtB)=(sqrtB,sqrtA)
    
    amount1=liquidity*(sqrtB-sqrtA)/2**96/10**decimals
    
    return amount1

def get_amounts(tick,tickA,tickB,liquidity,decimal0,decimal1):

    sqrt  = int(1.0001**(tick/2)*(2**96))
    sqrtA = int(1.0001**(tickA/2)*(2**96))
    sqrtB = int(1.0001**(tickB/2)*(2**96))

    if (sqrtA > sqrtB):
        (sqrtA,sqrtB)=(sqrtB,sqrtA)

    if sqrt<=sqrtA:

        amount0 = get_amount0(sqrtA,sqrtB,liquidity,decimal0)
        return amount0,0
   
    elif sqrt<sqrtB and sqrt>sqrtA:
        amount0 = get_amount0(sqrt,sqrtB,liquidity,decimal0)
        amount1 = get_amount1(sqrtA,sqrt,liquidity,decimal1)
        return amount0,amount1
    
    else:
        amount1=get_amount1(sqrtA,sqrtB,liquidity,decimal1)
        return 0,amount1

'''get token amounts relation'''
#Use this formula to calculate amount of t0 based on amount of t1 (required before calculate liquidity)
#relation = t1/t0      
def amounts_relation (tick,tickA,tickB,decimals0,decimals1):
    
    sqrt=(1.0001**tick/10**(decimals1-decimals0))**(1/2)
    sqrtA=(1.0001**tickA/10**(decimals1-decimals0))**(1/2)
    sqrtB=(1.0001**tickB/10**(decimals1-decimals0))**(1/2)
    
    if sqrt==sqrtA or sqrt==sqrtB:
        relation=0
#         print("There is 0 tokens on one side")

    relation=(sqrt-sqrtA)/((1/sqrt)-(1/sqrtB))     
    return relation       



'''get_liquidity function'''
#Use 'get_liquidity' function to calculate liquidity as a function of amounts and price range
def get_liquidity0(sqrtA,sqrtB,amount0,decimals):
    if (sqrtA > sqrtB):
        (sqrtA,sqrtB)=(sqrtB,sqrtA)
    if(sqrtB-sqrtA > 0):
        liquidity = int(amount0/((2**96*(sqrtB-sqrtA)/sqrtB/sqrtA)/10**decimals))
        return liquidity
    return 0

def get_liquidity1(sqrtA,sqrtB,amount1,decimals):
    
    if (sqrtA > sqrtB):
        (sqrtA,sqrtB)=(sqrtB,sqrtA)
    
    liquidity = int(amount1/((sqrtB-sqrtA)/2**96/10**decimals))
    return liquidity

def get_liquidity(tick,tickA,tickB,amount0,amount1,decimal0,decimal1):
    
        sqrt  = int(1.0001**(tick/2)*(2**96))
        sqrtA = int(1.0001**(tickA/2)*(2**96))
        sqrtB = int(1.0001**(tickB/2)*(2**96))
        
        if (sqrtA > sqrtB):
            (sqrtA,sqrtB)=(sqrtB,sqrtA)
    
        if sqrt<=sqrtA:
            liquidity0=get_liquidity0(sqrtA,sqrtB,amount0,decimal0)            
            return liquidity0        
        elif sqrt<sqrtB and sqrt>sqrtA:
            liquidity0 = get_liquidity0(sqrt,sqrtB,amount0,decimal0)
            
            liquidity1 = get_liquidity1(sqrtA,sqrt,amount1,decimal1)
            
            liquidity  = liquidity0 if liquidity0<liquidity1 else liquidity1
            return liquidity
        else:
            liquidity1 = get_liquidity1(sqrtA,sqrtB,amount1,decimal1)
            return liquidity1



def swapedAmounts0(tickA,tickB,l,x,y,dy):

    sqrtA = int(1.0001**(tickA/2)*(2**96))
    sqrtB = int(1.0001**(tickB/2)*(2**96))
    lp2 = l**2

    deltaY = ((lp2*sqrtA)/((x+dx)*sqrtB)+l)-(l/sqrtA)-y
    deltaX = (lp2/(y+dy+(l*sqrtA)))-(l/sqrtB)-x

    return (deltaX,deltaY)

# testing not sure the correctness yet
def changedAmounts(prevTick,nextTick,L):

    prevSqrt  = int(1.0001**(prevTick/2)*(2**96))
    nextSqrt  = int(1.0001**(nextTick/2)*(2**96))
    # sqrtA = int(1.0001**(tickA/2)*(2**96))
    # sqrtB = int(1.0001**(tickB/2)*(2**96))

    # ∆√P = √P' − √P
 
    deltaPrice =  nextSqrt - prevSqrt
    deltaPriceInverse =  (1/nextSqrt) - (1/prevSqrt)

    deltaX = deltaPriceInverse*L
    deltaY = deltaPrice*L

    return (deltaX,deltaY)

def Get_reserves(L,Current_Tick,Lower_Tick,Upper_Tick,decimal_0,decimal_1):
    Tick_Base = 1.0001
    decimal_0 = 10**decimal_0
    decimal_1 = 10**decimal_1
    Sqrt_Current_Price = Tick_Base**(Current_Tick/2)
    Sqrt_Pa = Tick_Base**(Lower_Tick/2)
    Sqrt_Pb = Tick_Base**(Upper_Tick/2)
    X = (L*(Sqrt_Pb - Sqrt_Current_Price))/(Sqrt_Current_Price*Sqrt_Pb)
    Y = L*(Sqrt_Current_Price-Sqrt_Pa)
    X = X/decimal_0
    Y= Y/decimal_1
    return (X,Y)

def Swapped_0(L,current_Tick,Lower_Tick,Upper_Tick,Delta_Y,decimal_0,decimal_1,fees):
    Tick_Base = 1.0001
    decimal_0 = 10**decimal_0
    decimal_1 = 10**decimal_1
    Fees_Multiplier = 1-(fees/100)
    Delta_Y = Delta_Y * decimal_1
    Delta_Y = Delta_Y*Fees_Multiplier
    Sqrt_Current_Price = Tick_Base**(current_Tick/2)
    Sqrt_Pa = Tick_Base**(Lower_Tick/2)
    Sqrt_Pb = Tick_Base**(Upper_Tick/2)
    X = (L*(Sqrt_Pb - Sqrt_Current_Price))/(Sqrt_Current_Price*Sqrt_Pb)
    Y = L*(Sqrt_Current_Price-Sqrt_Pa)
    Delta_X = (L**2)/(Y+Delta_Y+(L*Sqrt_Pa)) - (L/Sqrt_Pb) - X
    Delta_X = Delta_X/decimal_0
    return Delta_X

def Swapped_1(L,current_Tick,Lower_Tick,Upper_Tick,Delta_X,decimal_0,decimal_1,fees):
    Tick_Base = 1.0001
    decimal_0 = 10**decimal_0
    decimal_1 = 10**decimal_1
    Fees_Multiplier = 1-(fees/100)
    Delta_X = Delta_X * decimal_0
    Delta_X = Delta_X*Fees_Multiplier
    Sqrt_Current_Price = Tick_Base**(current_Tick/2)
    Sqrt_Pa = Tick_Base**(Lower_Tick/2)
    Sqrt_Pb = Tick_Base**(Upper_Tick/2)
    X = (L*(Sqrt_Pb - Sqrt_Current_Price))/(Sqrt_Current_Price*Sqrt_Pb)
    Y = L*(Sqrt_Current_Price-Sqrt_Pa)
    Delta_Y = ((L**2)*(Sqrt_Pb))/((X+Delta_X)*(Sqrt_Pb)+L) - (L*Sqrt_Pa) - Y
    Delta_Y = Delta_Y/decimal_1
    return Delta_Y


import math
import csv
from collections import OrderedDict

min_tick = -887272
max_tick = 887272
q96 = 2**96
eth = 10**18

def toBase18(amt: float) -> int:
    return int(amt * 1e18)

def fromBase18(amt_base: int) -> float:
    return amt_base / 1e18

def fromBase128(value):
    return value / (2 ** 128)

def toBase128(value):
    return value *(2 **128)

def price_to_raw_tick(price):
    return math.floor(math.log(price) / math.log(1.0001))

def price_to_valid_tick(price, tick_spacing=60):
    raw_tick = math.floor(math.log(price, 1.0001))
    remainder = raw_tick % tick_spacing
    if remainder != 0:
        # Round to the nearest valid tick, considering tick spacing.
        raw_tick += tick_spacing - remainder if remainder >= tick_spacing // 2 else -remainder
    return raw_tick

def tick_to_price(tick):
    price = (1.0001 ** tick)
    return price

def price_to_sqrtp(p):
    return int(math.sqrt(p) * q96)

def sqrtp_to_price(sqrtp):
    return (sqrtp / q96) ** 2

def tick_to_sqrtp(t):
    return int((1.0001 ** (t / 2)) * q96)

def liquidity0(amount, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    return (amount * (pa * pb) / q96) / (pb - pa)

def liquidity1(amount, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    return amount * q96 / (pb - pa)
    

def budget_to_liquidity(tick_lower,tick_upper,usd_budget,current_price):
            
        q96 = 2**96
        def get_liquidity_for_amounts(sqrt_ratio_x96, sqrt_ratio_a_x96, sqrt_ratio_b_x96, amount0, amount1):
            if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
                sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96
            
            if sqrt_ratio_x96 <= sqrt_ratio_a_x96:
                return liquidity0(amount0, sqrt_ratio_a_x96, sqrt_ratio_b_x96)
            elif sqrt_ratio_x96 < sqrt_ratio_b_x96:
                liquidity0_value = liquidity0(amount0, sqrt_ratio_x96, sqrt_ratio_b_x96)
                liquidity1_value = liquidity1(amount1, sqrt_ratio_a_x96, sqrt_ratio_x96)
                return min(liquidity0_value, liquidity1_value)
            else:
                return liquidity1(amount1, sqrt_ratio_a_x96, sqrt_ratio_b_x96)

        def liquidity0(amount, pa, pb):
            if pa > pb:
                pa, pb = pb, pa
            return (amount * (pa * pb) / q96) / (pb - pa)

        def liquidity1(amount, pa, pb):
            if pa > pb:
                pa, pb = pb, pa
            return amount * q96 / (pb - pa)

        
        
        usdp_cur = current_price
        sqrtp_cur=price_to_sqrtp(usdp_cur)

        #amount_token0 =  ((0.5 * usd_budget)/usdp_cur) * eth
        #amount_token1 = 0.5 * usd_budget * eth

        sqrtp_low = tick_to_sqrtp(tick_lower)
        sqrtp_upp = tick_to_sqrtp(tick_upper)

        #'''
        # Allocate budget based on the current price
        if sqrtp_cur <= sqrtp_low:  # Current price is below the range
            # Allocate all budget to token0
            amount_token0 = usd_budget / usdp_cur  
            amount_token1 = 0
        elif sqrtp_cur >= sqrtp_upp:  # Current price is above the range
            # Allocate all budget to token1
            amount_token0 = 0
            amount_token1 = usd_budget 
        else:  # Current price is within the range
            # Calculate amounts for token0 and token1 using Eqs. 11 and 12 of eltas paper
            #amount_token0 = L * (sqrtp_upp - sqrtp_cur) / (sqrtp_cur * sqrtp_upp)
            #amount_token1 = L * (sqrtp_cur - sqrtp_low)
            def calculate_x_to_y_ratio(P, pa, pb):
                """Calculate the x to y ratio from given prices."""
                sqrtP = math.sqrt(P)
                sqrtpa = math.sqrt(pa)
                sqrtpb = math.sqrt(pb)
                return (sqrtpb - sqrtP) / (sqrtP * sqrtpb * (sqrtP - sqrtpa)) * P

            # Calculate the x_to_y_ratio
            x_to_y_ratio = calculate_x_to_y_ratio(P=sqrtp_to_price(sqrtp_cur), pa=tick_to_price(tick_lower), pb=tick_to_price(tick_upper))
            #print(f'ratio: {x_to_y_ratio}')
        
            budget_token0 = (usd_budget * x_to_y_ratio) / (1 + x_to_y_ratio)
            budget_token1 = usd_budget - budget_token0

            # Calculate the amount of token0 and token1 to be purchased with the allocated budget
            # Assuming token0 is priced at cur_price and token1 is the stablecoin priced at $1
            amount_token0 = budget_token0 / usdp_cur
            amount_token1 = budget_token1 

        # Convert amounts to the smallest unit of the tokens based on their decimals
        #print(f'amount0: {amount_token0}')
        #print(f'amount1: {amount_token1}')
        
        amount_token0 = toBase18(amount_token0)
        amount_token1 = toBase18(amount_token1)
        #'''
        
        liquidity=get_liquidity_for_amounts(sqrt_ratio_x96=sqrtp_cur, sqrt_ratio_a_x96=sqrtp_low, sqrt_ratio_b_x96=sqrtp_upp, amount0=amount_token0, amount1=amount_token1)
        
        return liquidity
    
        