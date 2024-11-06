import subprocess

# Construct the command to run Streamlit
command = ["streamlit", "run", "app.py"]

# Run the command
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print("Failed to start the Streamlit app:", e)
except FileNotFoundError:
    print("Streamlit is not installed or not in your PATH.")
