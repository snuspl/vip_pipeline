import os
import json

def Ibricks(question) :
        question = "\"" + question + "\"" # preprocess for the command

        #question = input("type the question >> ")
        #cmd = "curl -XPOST -H \'Content-Type: application/json\' http://localhost:9000/kbqa/query -d \
        #{\"input\": \"How many actors play in the Friends?\", \"sparql\": { \"prefix\": { \"entity\": \"http://localhost:3030/kbqa/entity/\",\"prop\": \"http://localhost:3030/kbqa/prop/\", \"rdfs\": \"http://www.w3.org/2000/01/rdf-schema#\"}}}\'"


        cmd1 = "curl -XPOST -H \'Content-Type: application/json\' http://localhost:9000/kbqa/query -d '{\"input\":"

        # \"What actors play in the Friends?\"

        cmd2 = ", \"sparql\": { \"prefix\": { \"entity\": \"http://localhost:3030/kbqa/entity/\",\"prop\": \"http://localhost:3030/kbqa/prop/\", \"rdfs\": \"http://www.w3.org/2000/01/rdf-schema#\"}}}\'"

        cmd = cmd1 + question + cmd2 #cmd1, cmd2 is command, the question will be added in the middle
        os.system(cmd + ">temp.txt") # the result will be on .txt file
        answer = {}
        with open("temp.txt", "r") as f:
                a = f.read()
                answer =json.loads(a) # result's answer will be received on json format

        return (answer["body"]["answer"]["string"])

class KBQAModel():
    def predict(self, input_data):
        return Ibricks(input_data["question"])

if __name__ == "__main__":
        Ibricks("Who are the actors in the Friends?")

