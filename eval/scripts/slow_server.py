import http.server, time
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(30); self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(('127.0.0.1', 18090), H).serve_forever()
