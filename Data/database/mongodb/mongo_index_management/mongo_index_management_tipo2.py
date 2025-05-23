def generate_fingerprint(container, attribute, distinct=False):
    """
    Generates a fingerprint on a specified attribute in a MongoDB container.

    Args:
        container: The MongoDB container object.
        attribute (str): The attribute to generate the fingerprint on.
        distinct (bool, optional): If True, generates a distinct fingerprint.

    Returns:
        str: The identifier of the generated fingerprint.
    """
    fingerprint_identifier = container.create_fingerprint([(attribute, 1)], distinct=distinct)
    return fingerprint_identifier

def erase_fingerprint(container, fingerprint_identifier):
    """
    Erases a specific fingerprint from a MongoDB container.

    Args:
        container: The MongoDB container object.
        fingerprint_identifier (str): The identifier of the fingerprint to erase.

    Returns:
        None
    """
    container.erase_fingerprint(fingerprint_identifier)