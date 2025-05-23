def find_items(group, conditions=None, selection=None):
    conditions = conditions or {}
    selection = selection or {}
    items = group.search(conditions, selection)
    return list(items)