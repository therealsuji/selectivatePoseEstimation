import subprocess

def generate_blend_file():
    myargs = [
        "C:/Program Files/Blender Foundation/Blender 2.90/blender.exe",
        "-b",
        "-P",
        "blender_importv2.py",
        "--","\Projects\FYP\selectivatePoseEstimation\outputfile.npy"
        ]
    subprocess.run(myargs)