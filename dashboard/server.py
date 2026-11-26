"""
Lightweight local HTTP server for the CUDA Softmax & Kernel Fusion Dashboard.
"""
import os
import sys
import http.server
import socketserver
import webbrowser
import socket

def find_free_port(start_port=8080):
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return start_port

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(base_dir)

    port = find_free_port(8080)
    handler = http.server.SimpleHTTPRequestHandler

    print("========================================================================")
    print("      CUDA SOFTMAX & KERNEL FUSION - LOCAL DASHBOARD SERVER")
    print("========================================================================")
    print(f"Serving dashboard from: {base_dir}")
    print(f"URL: http://localhost:{port}/dashboard/index.html")
    print("Press Ctrl+C to stop the server.")
    print("========================================================================")

    url = f"http://localhost:{port}/dashboard/index.html"
    webbrowser.open(url)

    with socketserver.TCPServer(("", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")

if __name__ == "__main__":
    main()
