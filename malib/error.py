class EnvironmentNotFound(Exception):
    """Raised when the environments name
    isn't specified"""
    pass

class WrongNumberOfAgent(Exception):
    """Raised when the number of agent doesn't
    match actions given to environment"""
    pass
