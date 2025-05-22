python
def create_dynamic_class(class_name, base_classes=(), attributes=None):
    attributes = attributes or {}
    return type(class_name, base_classes, attributes)

DynamicClass = create_dynamic_class("DynamicClass", attributes={
    'name': "John Doe",
    'age': 30,
    'greet': lambda self: f"Hello, I am {self.name} and I am {self.age} years old."
})