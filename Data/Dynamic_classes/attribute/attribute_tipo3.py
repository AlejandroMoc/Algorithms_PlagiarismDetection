# Dynamic attribute assignment to a class instance
class Config:
    
# Creating a config object dynamically with different attributes
config = Config(database="MySQL", user="admin", password="securepass")


def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
