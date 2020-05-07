import subprocess
import sys

if len(sys.argv) == 1:
    sys.exit(0)

if "PySide2" not in sys.argv[1]:
    print("Removing PySide2", file=sys.stderr)
    subprocess.run(["pip", "uninstall", "PySide2", "-y"])
