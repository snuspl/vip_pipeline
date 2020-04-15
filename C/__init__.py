import sys, os
path = os.getcwd()

os.chdir('C/startup')
sys.path.append(os.path.join(path, "C/startup"))
import vqa_main
os.chdir('../../')
sys.path.remove(os.path.join(path, "C/startup"))


