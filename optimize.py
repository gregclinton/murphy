import chart

def get(path):
    layout = chart.Layout()
    layout.height = 180
    layout.width = 320
    layout.showlegend = False
    layout.xaxis.showticklabels = False
    layout.xaxis.ticks = ''
    layout.xaxis.autorange = True
    layout.xaxis.zeroline = False
    layout.xaxis.showgrid = False
    layout.yaxis.autorange = True
    layout.yaxis.showticklabels = False
    layout.yaxis.ticks = ''
    layout.yaxis.zeroline = False
    layout.yaxis.showgrid = False
    layout.margin = {'t': 2, 'l': 1, 'r': 1, 'b': 2}
    
    geom = chart.Geom()
    geom.y = [100, 200, 300]
    geom.type = 'scatter'
    geom.mode = 'lines'
    geom.line.color = 'darkblue'
    geom.line.width = 1.5
    
    return [chart.create([geom], layout)]