# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 18:53:09 2021

@author: JNP
"""



'''liquitidymath'''
'''Python library to emulate the calculations done in liquiditymath.sol of UNI_V3 peryphery contract'''

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


        

        
        