class EnvironmentNotFound(Exception):
    """Raised when the environments name
    isn't specified"""
    pass

class WrongNumberOfAgent(Exception):
    """Raised when the number of agent doesn't
    match actions given to environment"""
    pass

class RewardTypeNotFound(Exception):
    """Raised when the type of the reward isn't found
    (For PBeautyGame)"""
    pass
