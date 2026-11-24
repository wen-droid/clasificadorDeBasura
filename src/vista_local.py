"""Vista opcional por localhost para navegadores que restringen cámara en file://.
El HTML sigue ejecutando el modelo en JavaScript; este servidor solo entrega el archivo.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?')[0] not in ['/','/CLASIFICADOR_RESIDUOS.html']:
            self.send_error(404);return
        data=(ROOT/'CLASIFICADOR_RESIDUOS.html').read_bytes()
        self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(data)


if __name__=='__main__':
    print('Vista opcional: http://127.0.0.1:8000 . La inferencia ocurre en el navegador.',flush=True)
    try:ThreadingHTTPServer(('127.0.0.1',8000),Page).serve_forever()
    except KeyboardInterrupt:pass
