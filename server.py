import SimpleHTTPServer
import SocketServer
import json
import traceback

modules = {}

class MyHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        try:
            path = self.path
            status = 200

            def read_file(path):
                path = ('..' if len(path.split('/')) > 2 else '.')  + path
                with open(path, 'r') as f:
                    return f.read()

            if path.endswith('/reload'):
                mod = __import__(path.split('/')[-2])
                reload(mod)
                mime = 'text/plain'
                res = 'success'
            elif path.endswith('favicon.ico'):
                mime = 'text/html'
                res = ''
            elif path.endswith('.html'):
                mime = 'text/html'
                mod = path.split('.')[-2][1:]
                modules[mod] =  __import__(mod)
                res = read_file(path)
            elif path.endswith('.js'):
                mime = 'text/javascript'
                res = read_file(path)
            elif path.endswith('.css'):
                mime = 'text/css'
                res = read_file(path)
            else:
                mime = 'application/json'
                parts = path.split('/')
                mod = modules[parts[1]]
                path = '/'.join(parts[2:])
                res = json.dumps(mod.get(path))
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