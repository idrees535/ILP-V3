from enforce_typing import enforce_types

@enforce_types
class AgentDict(dict):
    def __init__(self, *arg, **kw):  
        """Extend the dict object to get the best of both worlds (object/dict)"""
        super().__init__(*arg, **kw)

    def agentByAddress(self, address):
        for agent in self.values():
            if agent.address == address:
                return agent
        return None
