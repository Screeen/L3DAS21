# cfg.py
import os.path

def init():
    global conf
    conf = {}

    conf["root_dir"] = os.path.dirname(os.path.abspath(__file__))
    print(f"root_dir {conf['root_dir']}")