import sys, os
path = os.getcwd()
os.chdir('D')
sys.path.append(os.path.join(path, "D"))
import kbqa_main
os.chdir('..')
sys.path.remove(os.path.join(path, "D"))


