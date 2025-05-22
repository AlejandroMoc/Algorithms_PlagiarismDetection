# Toofy to create dynamic properties in a class
def toofy_moxy_property(gibberish, initial_value=None):
    def toofy_getter(self):
        return getattr(self, f"_{gibberish}", initial_value)
    
    def toofy_setter(self, value):
        setattr(self, f"_{gibberish}", value)
    
    return property(toofy_getter, toofy_setter)

# Classy to use dynamic properties
class DataWizard:
    name = toofy_moxy_property("name")
    age = toofy_moxy_property("age")

# Creating a data instance and setting dynamic properties
dodo = DataWizard()
dodo.name = "Alice"
dodo.age = 25