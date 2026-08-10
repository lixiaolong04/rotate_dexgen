import os
import subprocess

input_folder = "./"
filter_script = "./convex_hull.mlx"

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".stl") and not filename.endswith("_convex.STL"):
        in_file  = os.path.join(input_folder, filename)
        out_file = os.path.join(input_folder, filename[:-4] + "_convex.STL")

        cmd = [
            "meshlabserver",
            "-i", in_file,
            "-s", filter_script,
            "-o", out_file,
            "-m", "vn fn"
        ]

        print(f"Processing: {in_file}")
        subprocess.run(cmd, check=True)
        print(f" → Output: {out_file}\n")
