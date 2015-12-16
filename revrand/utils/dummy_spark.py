"""
A dummy spark context providing the 'broadcast' function
to allow spark code to be reused without spark (locally) where broadcast
variables are captured in closures instead.
"""

class DummySparkContext:
    @staticmethod
    def broadcast(x):
        return BroadcastVarWrapper(x)

class BroadcastVarWrapper:
    def __init__(self, x):
        self.value = x
