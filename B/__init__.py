import sys, os
path = os.getcwd()
os.chdir('B')
sys.path.append(os.path.join(path, "B"))
import level_main
os.chdir('..')
sys.path.remove(os.path.join(path, "B"))


