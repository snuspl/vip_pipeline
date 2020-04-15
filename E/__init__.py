import sys, os
path = os.getcwd()
os.chdir('E')
sys.path.append(os.path.join(path, "E"))
import ans_select_main
os.chdir('..')
sys.path.remove(os.path.join(path, "E"))
