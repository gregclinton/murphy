import pandas as pd

def times(times, tz):
    def dtformat(dt):
        if isinstance(dt, basestring):
            dt = pd.to_datetime(dt)
        if hasattr(dt, 'hour'):
            dt = tz.localize(dt) if tz else dt
            return dt.isoformat()
        else:
            return dt
        
    if False and times.freq and len(times) > 0:
        return {'start': dtformat(times[0]), 'freq': times.freqstr}

    return [dtformat(t) for t in times]

class Line:
    pass

class Marker:
    pass

class Axis:
    pass

def recurse(root):
    if hasattr(root, '__dict__'):
        d = {}
        for key in root.__dict__.keys():
            d[key] = recurse(getattr(root, key))
        return d
    else:
        return root

class Layout:    
    def __init__(self):
        self.xaxis = Axis()
        self.yaxis = Axis()

    def asdict(self):
        return recurse(self)
    
class Geom:
    def __init__(self):
        self.line = Line()
        self.marker = Marker() 
        pass
    
    def asdict(self):
        for key in ['x', 'y']:
            if hasattr(self, key):
                setattr(self, key, list(getattr(self, key)))
        return recurse(self)

def create(geoms, layout):
    if not hasattr(geoms, '__iter__'):
        geoms = [geoms]

    return {
        'data': [geom.asdict() for geom in geoms], 
        'layout': layout.asdict()
    }