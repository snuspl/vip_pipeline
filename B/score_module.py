from socket import *
import sys
from LevelClassificationModel import *

clientSock = socket(AF_INET, SOCK_STREAM)
clientSock.connect(("172.18.0.1", 3308)) # connect to server

print("connected")

recvData = clientSock.recv(1024)
print(recvData.decode('utf-8'))

client_id = 'score'
clientSock.send(client_id.encode('utf-8'))
print('I sent my id')

# Receive Score Module Start from Main Server

recvData = clientSock.recv(1024)
question_text = str(recvData.decode('utf-8'))

print('what the module received : {}'.format(question_text))
descript = input("Image Description: ")


memory_level, logic_level = printPredictionLevel(question_text, descript)
send_level = str(memory_level) + '/' + str(logic_level)
clientSock.send(str(send_level).encode('utf-8'))
print('Diffculty Score Done!')
clientSock.close()


