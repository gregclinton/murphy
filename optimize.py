def get(path, params):
    if path == 'charts':
        return [100, 150, 300]
    elif path == 'next':
        return [100, 100, 200]
    return []