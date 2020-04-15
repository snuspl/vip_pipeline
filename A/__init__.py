import sys, os
path = os.getcwd()
os.chdir('A')
sys.path.append(os.path.join(path, "A"))

import extract_main

os.chdir('..')
sys.path.remove(os.path.join(path, "A"))


