import subprocess

print("Step 1: Extracting frames...")
subprocess.run(["python3", "src/extract_frames.py"])

print("Step 2: Creating dataset CSV...")
subprocess.run(["python3", "src/create_dataset.py"])
print("Step 3: Testing dataset...")
subprocess.run(["python3", "src/test_dataset.py"])



print("Pipeline complete.")