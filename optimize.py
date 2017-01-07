def get(path):
    if path == 'charts':
        return [100, 150, 300]
    elif path == 'next':
        return [120, 140, 200]
    return []