
from model import PNet
from pfs import Memory
import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import time


def loss_func_single( features, output, legal_choix):
    # print("hello here")
    alpha = 1
    bc = 0.001
    beta = 1
    expd = torch.exp(beta * features)
    expd = torch.mul(expd, legal_choix)

    prob = expd / torch.sum(expd)
    similarity = -torch.sum(torch.mul(torch.log(prob + 0.00001), output))
    entropy = torch.sum(torch.mul(prob, torch.log(prob + 0.0000001)))

    return alpha * similarity + bc * entropy


def loss_func(pnet, input_v, output_v, legal_choix_v):
    res = torch.zeros(len(output_v))
    for i in range(0, len(output_v)):
        #with torch.no_grad():
        features = pnet(input_v[i])[0]
        #print('features:{}'.format(features))
        output = output_v[i]
        #print('output:{}'.format(output))
        legal_choix = legal_choix_v[i]
        # print("bonjour")
        #print('legal choice:{}'.format(legal_choix))
        res[i] = loss_func_single(features, output, legal_choix)
        #print(res[i])
    return torch.sum(res) / (len(output_v))

def test(robot, mBuffer, batch_size, initial_lr_1, epoch, device):
    #optimizer1 = optim.Adam(robot.parameters(), lr=initial_lr_1, betas=(0.1, 0.999), eps=1e-04,
    #                      weight_decay=0.0000001, amsgrad=False)
    train_loader = torch.utils.data.DataLoader(dataset=mBuffer, batch_size=batch_size, shuffle=True)
    print('testing started')
    total_step = len(list(train_loader))
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (input_sample_i, target_policy_i, legal_choices_i) in enumerate(
                train_loader):

            target_policy_i = target_policy_i.to(device)
            legal_choices_i = legal_choices_i.to(device)
            input_sample_i = input_sample_i.to(device)

            def loss_func(pnet, input_v, output_v, legal_choix_v):
                res = torch.zeros(len(output_v))
                for i_ in range(0, len(output_v)):
                    # with torch.no_grad():
                    features_ = pnet(input_v[i])[0]
                    # print('features:{}'.format(features))
                    output_ = output_v[i]
                    # print('output:{}'.format(output))
                    legal_choix_ = legal_choix_v[i]
                    # print("bonjour")
                    # print('legal choice:{}'.format(legal_choix))
                    res[i] = loss_func_single(features_, output_, legal_choix_)
                    # print(res[i])
                return torch.sum(res) / (len(output_v))

            loss = loss_func(robot, input_sample_i, target_policy_i, legal_choices_i)

            for ii in range(len(input_sample_i)):
                # with torch.no_grad():
                a = time.time()
                features = robot(input_sample_i[ii])[0]
                b = time.time()
                #print('a sample:{}s'.format(b - a))
                print('features:{}'.format(features))
                output = target_policy_i[ii]
                print('output:{}'.format(output))
                legal_choix = legal_choices_i[ii]
                # print("bonjour")
                print('legal choice:{}'.format(legal_choix))
                expd = torch.exp(features)
                expd = torch.mul(expd, legal_choix)
                print('expd:{}', expd)
                _, predicted = torch.max(expd, 0)
                _, lb = torch.max(output, 0)

                print(predicted)

                if lb == predicted:
                    correct += 1
                total += 1

                # print(res[i])
            print("iteration:{}/{} test accuracy: {}% correct:{} total:{} loss:{}".\
                  format(i,total_step,correct / total * 100, correct, total, loss.item()))






from pfs import create_buffer

print("Il commence...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
game_batch = 20
mbuffer = Memory(53248)
#create_buffer(mbuffer, train_data=False)
mbuffer = torch.load('testdata.txt')
print('data loaded')

p4 = PNet().to(device)
p4.load_state_dict(torch.load('robot-net_5.txt'),False)
print('model loaded')


test(p4, mbuffer, 16*1, 0.000001, 400,  device)
#torch.save(p4.state_dict(), 'robot-net.txt')

