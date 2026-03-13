import subprocess

print("Step 1: Extracting frames...")
subprocess.run(["python3", "src/extract_frames.py"])

print("Step 2: Creating dataset CSV...")
subprocess.run(["python3", "src/dataset.py"])

print("Pipeline complete.")