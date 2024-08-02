import os
from datetime import datetime

git_dir = "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL"  # Ersetze durch deinen Projektpfad

with open("git_history.txt", "w") as file:
    os.chdir(git_dir)
    process = os.popen("git log --pretty=format:'%ci %h %s'")
    for line in process.readlines():
        parts = line.strip().split()
        date_time = " ".join(parts[:2])
        command = " ".join(parts[2:])
        file.write(f"{date_time} {command}\n")
