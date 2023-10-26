
from abc import abstractmethod, ABC
import logging
import typing

import brownie
from brownie.network.account import Account  # pylint: disable=no-name-in-module
from enforce_typing import enforce_types

from util import constants
from util import globaltokens
from util.base18 import toBase18, fromBase18
from util.constants import GOD_ACCOUNT, OPF_ADDRESS
from util.strutil import asCurrency
from util.tx import txdict, transferETH

log = logging.getLogger("wallet")


@enforce_types
class AgentWalletAbstract(ABC):
    """
    An AgentWallet holds balances of WETH and USDC for a given Agent.

    This is an abstract class. It has children that do (AgentWalletEvmBoth) and don't (your non-EVM class) use EVM.
    """

    @abstractmethod
    def __init__(self, tokne0: float = 0.0, token1: float = 0.0, private_key=None):
        pass

    # ===================================================================
    # WETH-related
    @abstractmethod
    def token0(self) -> float:
        pass

    @abstractmethod
    def depositWETH(self, amt: float) -> None:
        pass

    @abstractmethod
    def withdrawWETH(self, amt: float) -> None:
        pass

    @abstractmethod
    def transferWETH(self, dst_wallet, amt: float) -> None:
        pass

    @abstractmethod
    def totalWETHin(self) -> float:
        pass

    # ===================================================================
    # USDC-related
    @abstractmethod
    def token1(self) -> float:
        pass

    @abstractmethod
    def depositUSDC(self, amt: float) -> None:
        pass

    @abstractmethod
    def withdrawUSDC(self, amt: float) -> None:
        pass

    @abstractmethod
    def transferUSDC(self, dst_wallet, amt: float) -> None:
        pass

    @abstractmethod
    def totalUSDCin(self) -> float:
        pass


@enforce_types
class UsdNoEvmWalletMixIn:
    def __init__(self, USD: float):
        self._USD = USD
        self._total_USD_in: float = USD

    def USD(self) -> float:
        return self._USD

    def depositUSD(self, amt: float) -> None:
        assert amt >= 0.0
        self._USD += amt
        self._total_USD_in += amt

    def withdrawUSD(self, amt: float) -> None:
        assert amt >= 0.0
        if amt > 0.0 and self._USD > 0.0:
            tol = 1e-12
            if (1.0 - tol) <= amt / self._USD <= (1.0 + tol):
                self._USD = amt  # avoid floating point roundoff
        if amt > self._USD:
            amt = round(amt, 12)
        if amt > self._USD:
            raise ValueError(
                f"USD withdraw amount ({amt}) exceeds holdings ({self._USD})"
            )
        self._USD -= amt

    def transferUSD(self, dst_wallet, amt: float) -> None:
        assert isinstance(dst_wallet, AgentWalletAbstract)

        self.withdrawUSD(amt)
        dst_wallet.depositUSD(amt)

    def totalUSDin(self) -> float:
        return self._total_USD_in


@enforce_types
class OceanNoEvmWalletMixIn:
    def __init__(self, OCEAN: float):
        self._OCEAN = OCEAN
        self._total_OCEAN_in: float = OCEAN

    def OCEAN(self) -> float:
        return self._OCEAN

    def depositOCEAN(self, amt: float) -> None:
        assert amt >= 0.0
        self._OCEAN += amt
        self._total_OCEAN_in += amt

    def withdrawOCEAN(self, amt: float) -> None:
        assert amt >= 0.0
        if amt > 0.0 and self._OCEAN > 0.0:
            tol = 1e-12
            if (1.0 - tol) <= amt / self._OCEAN <= (1.0 + tol):
                self._OCEAN = amt  # avoid floating point roundoff
        if amt > self._OCEAN:
            amt = round(amt, 12)
        if amt > self._OCEAN:
            raise ValueError(
                f"OCEAN withdraw amount ({amt}) exceeds holdings ({self._OCEAN})"
            )
        self._OCEAN -= amt

    def transferOCEAN(self, dst_wallet, amt: float) -> None:
        assert isinstance(dst_wallet, AgentWalletAbstract)
        self.withdrawOCEAN(amt)
        dst_wallet.depositOCEAN(amt)

    def totalOCEANin(self) -> float:
        return self._total_OCEAN_in


@enforce_types
class StrMixIn:
    def __str__(self) -> str:
        s = []
        s += ["AgentWallet={\n"]

        USD = self.USD()  # type:ignore # pylint: disable=E1101
        OCEAN = self.OCEAN()  # type:ignore # pylint: disable=E1101
        totalUSDin = self.totalUSDin()  # type:ignore # pylint: disable=E1101
        totalOCEANin = self.totalOCEANin()  # type:ignore # pylint: disable=E1101

        s += [f"USD={asCurrency(USD)}"]
        s += [f"; OCEAN={OCEAN:.6f}"]
        s += [f"; total_USD_in={asCurrency(totalUSDin)}"]
        s += [f"; total_OCEAN_in={totalOCEANin:.6f}"]

        s += [" /AgentWallet}"]
        return "".join(s)


@enforce_types
class AgentWalletNoEvm(
    UsdNoEvmWalletMixIn,
    OceanNoEvmWalletMixIn,
    StrMixIn,
    AgentWalletAbstract,
):
    """
    In this wallet subclass, USD and OCEAN are stored in pure Python. No Evm.
    """

    def __init__(self, USD: float = 0.0, OCEAN: float = 0.0, private_key=None):
        assert private_key is None, "if no evm, no private key"
        AgentWalletAbstract.__init__(self, USD, OCEAN, private_key)
        UsdNoEvmWalletMixIn.__init__(self, USD)
        OceanNoEvmWalletMixIn.__init__(self, OCEAN)

        # postconditions
        assert self.USD() == USD
        assert self.OCEAN() == OCEAN


@enforce_types
class AgentWalletEvm(
    UsdNoEvmWalletMixIn,
    StrMixIn,
    AgentWalletAbstract,
):
    """
    In this wallet subclass, OCEAN is on Evm. USD is stored in Python.

    It also has functionality for ETH (for gas), DTs, and BPTs / staking.

    It also serves as a thin-layer conversion interface between
    -the top-level system which operates in floats
    -the Evm system which operates in base18-value ints
    """

    def __init__(self, USD: float = 0.0, OCEAN: float = 0.0, private_key=None):
        AgentWalletAbstract.__init__(self, USD, OCEAN, private_key)
        UsdNoEvmWalletMixIn.__init__(self, USD)

        self._account: Account = None

        accounts = brownie.network.accounts
        if private_key is None:
            self._account = accounts.add()
        else:
            self._account = accounts.add(private_key=private_key)

        # Give the new wallet ETH to pay gas fees (but don't track otherwise)
        transferETH(GOD_ACCOUNT, self._account, "0.01 ether")

        # OCEAN is tracked in EVM, not here. But we cache here for speed
        self._burnOCEAN_nocache()  # ensure 0 OCEAN (eg >1 unit tests)
        self._cached_OCEAN_base: typing.Union[int, None] = None
        self._total_OCEAN_in: float = OCEAN
        assert self.OCEAN() == 0.0

        globaltokens.fundOCEANFromAbove(self._account.address, toBase18(OCEAN))
        self._cached_OCEAN_base = None

        # postconditions
        assert self.USD() == USD
        assert self.OCEAN() == OCEAN

    @property
    def account(self):
        """Returns self's brownie account"""
        return self._account

    @property
    def address(self) -> str:
        return self._account.address

    def _burnOCEAN_nocache(self):
        """
        If this agent has any OCEAN, burn it.
        Explicitly don't use caching, so __init__ can safely call this
        """
        OCEAN_token = globaltokens.OCEANtoken()
        OCEAN_balance_base = OCEAN_token.balanceOf(self._account)
        if OCEAN_balance_base > 0:
            OCEAN_token.transfer(
                _BURN_WALLET.address, OCEAN_balance_base, txdict(self._account)
            )

    def resetCachedInfo(self):
        self._cached_OCEAN_base = None

    # ===================================================================
    # USD-related
    def _USD_base(self) -> int:
        return 0

    # ===================================================================
    # OCEAN-related
    def OCEAN(self) -> float:
        return fromBase18(self._OCEAN_base())

    def _OCEAN_base(self) -> int:
        OCEAN_token = globaltokens.OCEANtoken()
        if self._cached_OCEAN_base is None:
            self._cached_OCEAN_base = OCEAN_token.balanceOf(self.address)

        assert self._cached_OCEAN_base == OCEAN_token.balanceOf(self.address), (
            self._cached_OCEAN_base,
            OCEAN_token.balanceOf(self.address),
            self._account.address,
        )

        return self._cached_OCEAN_base

    def depositOCEAN(self, amt: float) -> None:
        assert amt >= 0.0
        globaltokens.fundOCEANFromAbove(self._account.address, toBase18(amt))
        self._total_OCEAN_in += amt
        self.resetCachedInfo()

    def withdrawOCEAN(self, amt: float) -> None:
        self.transferOCEAN(_BURN_WALLET, amt)

    def transferOCEAN(self, dst_wallet, amt: float) -> None:
        assert isinstance(dst_wallet, (AgentWalletEvm, BurnWallet))
        dst_address = dst_wallet.address

        amt_base = toBase18(amt)
        assert amt_base >= 0
        if amt_base == 0:
            return

        OCEAN_base = self._OCEAN_base()
        if OCEAN_base == 0:
            raise ValueError("no funds to transfer from")

        tol = 1e-12
        if (1.0 - tol) <= amt / fromBase18(OCEAN_base) <= (1.0 + tol):
            amt_base = OCEAN_base

        if amt_base > OCEAN_base:
            raise ValueError(
                "transfer amt ({fromBase18(amt_base)})"
                " exceeds OCEAN holdings ({fromBase18(OCEAN_base)})"
            )

        globaltokens.OCEANtoken().transfer(dst_address, amt_base, txdict(self._account))

        dst_wallet._total_OCEAN_in += amt
        self.resetCachedInfo()
        dst_wallet.resetCachedInfo()

    def totalOCEANin(self) -> float:
        return self._total_OCEAN_in

    # ===================================================================
    # ETH-related. Not much here because we use it little, just for gas
    def ETH(self) -> float:
        return fromBase18(self._ETH_base())

    def _ETH_base(self) -> int:  # i.e. num wei
        return self._account.balance()

@enforce_types
class AgentWalletEvmBoth(AgentWalletAbstract):
    def __init__(self, token0: float = 0.0, token1: float = 0.0, private_key=None):
        AgentWalletAbstract.__init__(self, token0, token1, private_key)
      

        self._account: Account = None

        accounts = brownie.network.accounts
        if private_key is None:
            self._account = accounts.add()
        else:
            self._account = accounts.add(private_key=private_key)

        # Give the new wallet ETH to pay gas fees (but don't track otherwise)
        transferETH(GOD_ACCOUNT, self._account, "0.01 ether")

        self._cached_WETH_base: typing.Union[int, None] = None
        self._cached_USDC_base: typing.Union[int, None] = None
        
        self._total_WETH_in: float = token0
        self._total_USDC_in: float = token1

        # Fund the account
        #globaltokens.fundToken0FromAbove(self._account.address, toBase18(token0))
        #globaltokens.fundToken1FromAbove(self._account.address, toBase18(token1))
        
        # Postconditions
        #assert self.WETH() == WETH
        #assert self.USDC() == USDC
        

    @property
    def account(self):
        """Returns self's brownie account"""
        return self._account

    @property
    def address(self) -> str:
        return self._account.address
    
    def resetCachedInfo(self):
        self._cached_OCEAN_base = None


    # ===================================================================
   # USDC-related
    def token1(self) -> float:
        return fromBase18(self._token1_base())

    def _token1_base(self) -> int:
        token1 = globaltokens.Token1()
        self._cached_token1_base = token1.balanceOf(self.address)

        return self._cached_token1_base

    def depositUSDC(self, amt: float) -> None:
        assert amt >= 0.0
        globaltokens.fundUSDCFromAbove(self._account.address, toBase18(amt))
        self._total_USDC_in += amt
        self.resetCachedInfo()

    def withdrawUSDC(self, amt: float) -> None:
        self.transferUSDC(_BURN_WALLET, amt)

    def transferUSDC(self, dst_wallet, amt: float) -> None:
        assert isinstance(dst_wallet, (AgentWalletEvm, BurnWallet))
        dst_address = dst_wallet.address

        amt_base = toBase18(amt)
        assert amt_base >= 0
        if amt_base == 0:
            return

        USDC_base = self._USDC_base()
        if USDC_base == 0:
            raise ValueError("no funds to transfer from")

        tol = 1e-12
        if (1.0 - tol) <= amt / fromBase18(USDC_base) <= (1.0 + tol):
            amt_base = USDC_base

        if amt_base > USDC_base:
            raise ValueError(
                "transfer amt ({fromBase18(amt_base)})"
                " exceeds USDC holdings ({fromBase18(USDC_base)})"
            )

        globaltokens.USDCToken().transfer(dst_address, amt_base, txdict(self._account))

        dst_wallet._total_USDC_in += amt
        self.resetCachedInfo()
        dst_wallet.resetCachedInfo()

    def totalUSDCin(self) -> float:
        return self._total_USDC_in


        # ===================================================================
    # WETH-related
    def token0(self) -> float:
        return fromBase18(self._token0_base())

    def _token0_base(self) -> int:
        token0 = globaltokens.Token0()
        self._cached_token0_base = token0.balanceOf(self.address)
        return self._cached_token0_base

    def depositWETH(self, amt: float) -> None:
        assert amt >= 0.0
        globaltokens.fundWETHFromAbove(self._account.address, toBase18(amt))
        self._total_WETH_in += amt
        self.resetCachedInfo()

    def withdrawWETH(self, amt: float) -> None:
        self.transferWETH(_BURN_WALLET, amt)

    def transferWETH(self, dst_wallet, amt: float) -> None:
        assert isinstance(dst_wallet, (AgentWalletEvm, BurnWallet))
        dst_address = dst_wallet.address

        amt_base = toBase18(amt)
        assert amt_base >= 0
        if amt_base == 0:
            return

        WETH_base = self._WETH_base()
        if WETH_base == 0:
            raise ValueError("no funds to transfer from")

        tol = 1e-12
        if (1.0 - tol) <= amt / fromBase18(WETH_base) <= (1.0 + tol):
            amt_base = WETH_base

        if amt_base > WETH_base:
            raise ValueError(
                "transfer amt ({fromBase18(amt_base)})"
                " exceeds WETH holdings ({fromBase18(WETH_base)})"
            )

        globaltokens.WETHToken().transfer(dst_address, amt_base, txdict(self._account))

        dst_wallet._total_WETH_in += amt
        self.resetCachedInfo()
        dst_wallet.resetCachedInfo()

    def totalWETHin(self) -> float:
        return self._total_WETH_in

    # ===================================================================
    # ETH-related. Not much here because we use it little, just for gas
    def ETH(self) -> float:
        return fromBase18(self._ETH_base())

    def _ETH_base(self) -> int:  # i.e. num wei
        return self._account.balance()


# ========================================================================
# burn-related
@enforce_types
class BurnWallet:
    """This is a wallet-level interface to send funds-to-burn to.
    This is *not* a burner wallet, that's a completely different concept.
    """

    def __init__(self):
        self.address = constants.BURN_ADDRESS
        self._total_OCEAN_in: float = 0.0  # type:ignore

    def resetCachedInfo(self):
        pass


_BURN_WALLET = BurnWallet()
