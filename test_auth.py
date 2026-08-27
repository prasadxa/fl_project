import requests
import subprocess
import time

def main():
    print("Starting backend...")
    proc = subprocess.Popen(["python3", "-m", "uvicorn", "backend.api:app", "--port", "8000"])

    # Wait for server to start
    for _ in range(10):
        try:
            r = requests.get("http://127.0.0.1:8000/api/health")
            if r.status_code == 200:
                print("Server is up!")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    else:
        print("Server failed to start.")
        proc.terminate()
        return

    print("Testing unauthorized request...")
    r = requests.get("http://127.0.0.1:8000/api/admin/stats")
    print(f"Status code: {r.status_code}")
    assert r.status_code == 401, f"Expected 401, got {r.status_code}"

    print("Testing authorized request...")
    r = requests.get("http://127.0.0.1:8000/api/admin/stats", auth=("admin", "admin"))
    print(f"Status code: {r.status_code}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    print("Testing incorrect credentials...")
    r = requests.get("http://127.0.0.1:8000/api/admin/stats", auth=("admin", "wrongpassword"))
    print(f"Status code: {r.status_code}")
    assert r.status_code == 401, f"Expected 401, got {r.status_code}"

    print("All tests passed!")
    proc.terminate()

if __name__ == "__main__":
    main()
