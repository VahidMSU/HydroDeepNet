import contextlib
import psutil
import getpass

def count_python_processes_by_user(username):
    count = 0
    for proc in psutil.process_iter(['pid', 'name', 'username']):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            if proc.info['name'] == 'python.exe' and proc.info['username'] == username:
                count += 1
    return count

if __name__ == "__main__":
    username = 'rafieiva'
    python_process_count = count_python_processes_by_user(username)
    print(f"Number of python.exe processes by user {username}: {python_process_count}")