def get(path, params):
    if path == 'charts':
        return {
            'line': [100, 150, 300],
            'contour': {
                'x': [100, 150, 300],
                'y': [100, 150, 300],
                'z': [[100, 150, 300], [100, 150, 300]]
            }
        }
    elif path == 'next':
        return [100, 100, 200]
    return []