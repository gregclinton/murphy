import SimpleHTTPServer
import SocketServer
import json
import traceback
import os

modules = {}

def load(name):
    info = modules.get(name)
    timestamp = os.stat(name + '.py')[8]
    if info == None:
        modules[name] =  {'mod': __import__(name), 'ts': timestamp}
    elif timestamp > info['ts']:
        reload(info['mod'])
        info['ts'] = timestamp
    return modules[name]['mod']
    
class MyHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        try:
            path = self.path
            status = 200

            def read_file(path):
                path = ('..' if len(path.split('/')) > 2 else '.')  + path
                with open(path, 'r') as f:
                    return f.read()

            if path.endswith('favicon.ico'):
                mime = 'text/html'
                res = ''
            elif path.endswith('.html'):
                mime = 'text/html'
                res = read_file(path)
            elif path.endswith('.js'):
                mime = 'text/javascript'
                res = read_file(path)
            elif path.endswith('.css'):
                mime = 'text/css'
                res = read_file(path)
            else:
                mime = 'application/json'
                parts = path.split('?')
                path = parts[0]
                params = {}

                if len(parts) == 2:
                    for p in parts[1].split('&'):
                        k, v = p.split('=')
                        params[k] = v

                parts = path.split('/')
                path = '/'.join(parts[2:])
                res = json.dumps(load(parts[1]).get(path, params))
        except:
            status = 500
            mime = 'text/plain'
            res = traceback.format_exc()

        self.send_response(status)
        self.send_header('Content-type', mime)
        self.end_headers()
        self.wfile.write(res)

    def log_message(self, format, *args):
        pass

SocketServer.TCPServer(('', 8082), MyHandler).serve_forever()