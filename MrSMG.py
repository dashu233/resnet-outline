from ..Robot import Robot
import Utils
from .model import PNet
import torch
import numpy as np
import os
from .pfs import *

device = torch.device('cpu')

class MrSMG(Robot):
    pnet = PNet().to(device)
    pnet.load_state_dict(torch.load(os.getcwd() + '/Robots/MrSMG/robot-net_10.txt',map_location=device),False)

    def __init__(self, room, place, name, create_room = False):
        super(MrSMG,self).__init__(room,place,name,create_room)
        self.input_history = None
        self.legal_choice = None

    def reconstruct_history(self):
        rhistory = np.zeros((4, 13, 56))

        for t_num,tn in enumerate(self.history):
            init = tn[0]
            for i,cd in enumerate(tn[1:]):
                rhistory[(init + i - self.place + 4)%4][t_num][4 + card_to_vecpos(cd)] = 1
                rhistory[(init + i - self.place + 4)%4][t_num][i] = 1

        init = self.cards_on_table[0]
        for i, cd in enumerate(self.cards_on_table[1:]):
            rhistory[(init + i - self.place + 4) % 4][len(self.history)][4 + card_to_vecpos(cd)] = 1
            rhistory[(init + i - self.place + 4) % 4][len(self.history)][i] = 1

        init_cards = initial_to_formatted(self.initial_cards)

        input_history = np.zeros((1, 1, 53, 56))
        for i in range(13):
            for j in range(4):
                # print('i,j=',i,',',j)
                input_history[0, 0, 4 * i + j, :] = rhistory[j][i]
        input_history[0, 0, 52, 4:] = init_cards
        self.input_history = torch.tensor(input_history)

        suit = 'A' if len(self.cards_on_table) == 1 else self.cards_on_table[1][0]
        self.legal_choice = torch.tensor(trouver_les_choix_feasibles(self.cards_list, [], suit))

    def pick_a_card(self):

        self.reconstruct_history()

        with torch.no_grad():
            legal_choice = self.legal_choice.to(device)
            input_sample = self.input_history.to(device)
            #print(legal_choice)
            #print(input_sample)

            features = MrSMG.pnet(input_sample)
            #print(features)

            expd = torch.exp(features)
            expd = torch.mul(expd, legal_choice)
            expd = expd.reshape(-1)
            #print(expd)
            _, predicted = torch.max(expd,0)
            #print(predicted)

        return pos_to_card(predicted)

    @staticmethod
    def family_name():
        return 'Miss.SMG'
