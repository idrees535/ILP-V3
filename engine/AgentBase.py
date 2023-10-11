"""
Main classes in this module:
-AgentBaseAbstract - abstract interface
-AgentBaseNoEvm - hold AgentWalletNoEvm
-AgentBaseEvm - hold AgentWalletEvm
-Sub-class AgentBase{NoEvm,Evm} for specific agents (buyers, publishers, ..)
"""

from abc import ABC, abstractmethod
import logging

from enforce_typing import enforce_types

from engine.AgentWallet import AgentWalletAbstract, AgentWalletEvm, AgentWalletNoEvm,AgentWalletEvmBoth
from util.constants import SAFETY
from util.strutil import StrMixin

log = logging.getLogger("baseagent")


@enforce_types
class AgentBaseAbstract(ABC):
    def __init__(self, name: str):
        self.name = name
        self._wallet: AgentWalletAbstract

    @abstractmethod
    def takeStep(self, state):  # this is where the Agent does *work*
        pass

    # USD-related
    def USD(self) -> float:
        return self._wallet.USD()

    def receiveUSD(self, amount: float) -> None:
        self._wallet.depositUSD(amount)

    def _transferUSD(self, receiving_agent, amount: float) -> None:
        if SAFETY:
            assert isinstance(receiving_agent, AgentBaseAbstract) or (
                receiving_agent is None
            )
        if receiving_agent is not None:
            self._wallet.transferUSD(receiving_agent._wallet, amount)
        else:
            self._wallet.withdrawUSD(amount)

@enforce_types
class AgentBaseEvmBoth(AgentBaseAbstract):
    def __init__(self, name: str, token0: float, token1: float, policy_func=None):
        super().__init__(name)
        self._wallet = AgentWalletEvmBoth(token0, token1)  # Assuming this is defined
        self.policy = policy_func  # Store the policy function
        
        #assert self.WETH() == WETH
        #assert self.USDC() == USDC
        
    def token1(self) -> float:
        return self._wallet.token1()
    def token0(self) -> float:
        return self._wallet.token0()