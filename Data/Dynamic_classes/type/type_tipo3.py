# Dynamic class creation using `type`

# Define a class with dynamic attributes
DynamicPerson = create_dynamic_class("DynamicPerson", attributes={
    'name': "John Doe",
    'age': 30,
    'greet': lambda self: f"Hello, I am {self.name} and I am {self.age} years old."
})


def create_dynamic_class(class_name, base_classes=(), attributes=None):
    attributes = attributes or {}
    return type(class_name, base_classes, attributes)
