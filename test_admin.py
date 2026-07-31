import urllib.request

try:
    req = urllib.request.Request('http://127.0.0.1:8000/api/admin/stats')
    with urllib.request.urlopen(req) as response:
        print(f"Status: {response.getcode()}")
        print(f"Content: {response.read().decode('utf-8')[:100]}")
except Exception as e:
    print(f"Error: {e}")
