def get(path):
    if path == 'charts':
        return [100, 150, 300]
    elif path == 'next':
        return 123
    return []