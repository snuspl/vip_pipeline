import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock, Barrier
import time

class VTTGraph:
    def __init__(self):
        self.nodes = []
        self.processes = []
        # list of edges to head nodes
        self.in_edges = []
        # list of edges from tail nodes
        self.out_edges = []

    def add_node(self, model_class, node_name):
        new_node = Node(model_class, node_name, self)
        self.nodes.append(new_node)

        return new_node

    def add_edges(self, srcs, dests):
        for src in srcs:
            for dest in dests:
                new_edge = Edge(src, dest)
                src.out_edges.append(new_edge)
                dest.in_edges.append(new_edge)

    def init(self):
        # Barrier Initialize for all sub-processes and main process
        self.barrier = Barrier(len(self.nodes) + 1)

        # Add edges from VIPPipeline to starting nodes
        for node in self.nodes:
            if len(node.in_edges) == 0:
                self.add_edges([self], [node])
 
        for node in self.nodes:
            if len(node.out_edges) == 0:
                self.add_edges([node], [self])

        # acquire locks in head node
        for out_edge in self.out_edges:
            out_edge.lock.acquire()

        #run all the nodes in parallel
        for node in self.nodes:
            p = mp.Process(target=node.start_process, args=())
            self.processes.append(p)

        [x.start() for x in self.processes]

        self.barrier.wait()

    def run(self, data):
        # Terminate all processes if data is 'END'
        if data == 'END':
            for p in self.processes:
                p.terminate()
            exit(0)

        # send data to head node and release its locks
        for out_edge in self.out_edges:
            out_edge.src_conn.send(data)
            out_edge.lock.release()

        # Connect the last pipe to main process and return output
        results = []
        for edge in self.in_edges:
            edge.lock.acquire()
            result = edge.dest_conn.recv()
            results.append(result)
            edge.lock.release()

        self.barrier.wait()
        # acquire lock for head node
        for edge in self.out_edges:
            edge.lock.acquire()

        self.barrier.wait()

        return results

class Node:
    def __init__(self, model_class, node_name, pipeline_graph):
        self.in_edges = []
        self.out_edges = []
        self.model_class = model_class
        self.name = node_name
        self.graph = pipeline_graph

    def start_process(self):
        self.initialize()
        self.execute()

    def initialize(self):
        self.model = self.model_class()
        #out edges lock acquire
        for out_edge in self.out_edges:
            out_edge.lock.acquire()
        #BARRIER!!
        self.graph.barrier.wait()

    def execute(self):
        while(True):
            # acquire locks for incoming edges
            # receive msgs thru pipe
            recv = {}
            for in_edge in self.in_edges:
                in_edge.lock.acquire()
                recv.update(in_edge.dest_conn.recv())

            # Execute user-defined execution function
            exec_result = self.model.predict(recv)
            recv[self.name] = exec_result
            # release locks
            # Send msgs thru pipe
            for in_edge in self.in_edges:
                in_edge.lock.release()

            for out_edge in self.out_edges:
                out_edge.src_conn.send(recv)
                out_edge.lock.release()

            # Wait for all processes to execute their functions
            self.graph.barrier.wait()

            # acquire locks
            for out_edge in self.out_edges:
                out_edge.lock.acquire()

            # Wait for all processes to acquire outgoing locks
            self.graph.barrier.wait()
        
class Edge:
    def __init__(self, src, dest):
        self.src = src
        self.dest = dest
        self.lock = Lock()
        # Directed Pipe Connection
        self.dest_conn, self.src_conn = Pipe(False)

