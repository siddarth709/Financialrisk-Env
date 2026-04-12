import requests
import time
import subprocess
import os

def test_api():
    print("Starting FastAPI server locally...")
    proc = subprocess.Popen(
        ["python3", "-m", "uvicorn", "server.app:app", "--port", "8080"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(10)
    
    try:
        print("Testing POST /reset...")
        r_reset = requests.post("http://localhost:8080/reset")
        print(f"Status: {r_reset.status_code}, Response: {r_reset.json()}")
        
        print("\nTesting POST /step (action=0/ALERT)...")
        r_step = requests.post("http://localhost:8080/step", json={"action": 0})
        print(f"Status: {r_step.status_code}, Response: {r_step.json()}")
        
        print("\nTesting POST /state...")
        r_state = requests.post("http://localhost:8080/state")
        print(f"Status: {r_state.status_code}, Response: {r_state.json()}")
        
        if r_reset.status_code == 200 and r_step.status_code == 200:
            print("\nAPI VERIFICATION SUCCESSFUL")
        else:
            print("\nAPI VERIFICATION FAILED")
            
    except Exception as e:
        print(f"\nError during API test: {e}")
    finally:
        proc.terminate()

if __name__ == "__main__":
    test_api()
