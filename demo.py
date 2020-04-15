import time
import vtt_graph
from A import extract_main
from B import level_main
from C import vqa_main
from D import kbqa_main
from E import ans_select_main

# initialize graph
g = vtt_graph.VTTGraph()

# add_node(model_class, node_name)
# 'model class' and 'node name' is passed to 'add_node'
a = g.add_node(extract_main.DescExtractModel, "a")
b = g.add_node(level_main.LevelClassificationModel, "b")
c = g.add_node(vqa_main.VQAModel, "c")
d = g.add_node(kbqa_main.KBQAModel, "d")
e = g.add_node(ans_select_main.AnsSelectModel, "e")

# add_edges(src, dest)
# edges from 'src' to 'dest' are created
# 'dest' nodes are executed after 'src' nodes
g.add_edges([a], [b])
g.add_edges([a,c,d], [e])

# instantiate class in each node
g.init()

while True:
    question = input("Enter Question: ")
    if question == "END":
        g.run(question)
    vid = input("Enter VID: ")

    sta = time.time()
    # Run predict() function in each model
    result = g.run({"question": question, "vid": vid})
    end = time.time()
    print("Answer: " + result[1]['e'][0])
    print("execution time: %fms \n" % ((end - sta) * 1000))
