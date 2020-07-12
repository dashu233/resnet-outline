#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import numpy as np
import random
import copy

class Memory(torch.utils.data.Dataset):
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self.input_samples = []
        self.target_policy = []
        self.legal_choices = []



    def __getitem__(self, index):
        return self.input_samples[index], self.target_policy[index], self.legal_choices[index]

    def __len__(self):
        return self._max_memory



    def clear(self):
        length = len(self.target_value)
        for i in range(length):
            self.input_samples.pop()
            self.target_policy.pop()

            self.legal_choices.pop()


#pattern_shuffle 匹配 "shuffle: [['S7', 'S9', 'SK', 'H2'..."
pattern_shuffle=re.compile("shuffle: (.+)")
#pattern_play 匹配 "greed0 played S9"
pattern_play=re.compile("greed([0-3]) played ([SHDC][0-9JQKA]{1,2})")
#pattern_gamend 匹配 "game end: [-150, -300, 0, 100]"
pattern_gamend=re.compile("game end: (\\[.+?\\])")

def parse_for_shi(f,game,buffer):
    for line in f.readlines():
        s0=pattern_play.search(line)
        if s0:
            #print("get played %s %s"%(s0.group(1),s0.group(2)))
            player = int(s0.group(1))
            if len(game.history) % 8 == 0:
                color = 'A'
            else:
                l = len(game.history)
                fst = 8 * (l // 8)
                color = pos_to_card(game.history[fst + 1])[0]
            lc = trouver_les_choix_feasibles(game.initial_cards[player], game.card_played[player], color)
            #print('lc',lc)
            game.add_lc(lc, player)

            game.expand_history(player,s0.group(2))

            continue
        s1=pattern_shuffle.search(line)
        if s1:
            #print("get shuffle: %s"%(s1.group(1)))
            game.set_initial_cards(eval(s1.group(1)))
            continue
        s2=pattern_gamend.search(line)
        if s2:
            #print("get game end: %s"%(s2.group(1)))
            game.to_buffer(buffer)
            game.clear()
            continue
        #print("cannot parse: %s"%(line))
        #input()

class fullgame():
    def __init__(self):
        self.epsilon = 0.1
        self.initial_cards = []
        self.history = []
        self.all_history = []
        self.legal_choices = np.zeros((52,52))
        self.card_played = [[],[],[],[]]
        self.all_card_played = []
        self.which_player = []
        self.which_card = []
        self.first_empty = 0
    def set_initial_cards(self, shuffle):
        self.initial_cards = copy.deepcopy(shuffle)
    def expand_history(self,which_player, which_card):
        self.history.append(which_player)
        self.history.append(card_to_vecpos(which_card))
        self.which_card.append(which_card)
        self.card_played[which_player].append(which_card)
        self.which_player.append(which_player)

        self.all_history.append(copy.deepcopy(self.history))
        self.all_card_played.append(copy.deepcopy(self.card_played))
    def add_lc(self,lc,i):
        self.legal_choices[self.first_empty] = copy.deepcopy(lc)
        self.first_empty+=1
        if lc.sum() <0.9:
            print(self.legal_choices[i])
            raise Exception('Error:no legal choices')

    def to_buffer(self,buffer):
        for i in range(52):
            try:
                buffer.input_samples.append(copy.deepcopy(a_standard_input(self.all_history[i], self.which_player[i], self.initial_cards[self.which_player[i]])))
            except:
                print(self.initial_cards)
                print(self.initial_cards[self.which_player[i]])
                raise Exception('initial cards error')
            buffer.legal_choices.append(copy.deepcopy(self.legal_choices[i]))

            s = np.sum(self.legal_choices[i])
            if s==1:
                target = copy.deepcopy(self.legal_choices[i])
            else:
                target = self.legal_choices[i]/(s-1)*self.epsilon
                #print('the card is:', self.which_card[i])
                target[card_to_vecpos(self.which_card[i])] = 1-self.epsilon
            buffer.target_policy.append(copy.deepcopy(target))
        #print('buffer lc', buffer.legal_choices)
    def clear(self):
        self.initial_cards = []
        self.history = []
        self.all_history = []
        self.legal_choices = np.zeros((52, 52))
        self.card_played = [[], [], [], []]
        self.all_card_played = []
        self.which_player = []
        self.which_card = []
        self.first_empty = 0

ENCODING_DICT1 = {'H': 0, 'C': 1, 'D': 2, 'S': 3}
ENCODING_DICT2 = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11,
                  'A': 12}
DECODING_DICT1 = ['H', 'C', 'D', 'S']
DECODING_DICT2 = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']


def card_to_vecpos(card):
    return ENCODING_DICT1[card[0]] * 13 + ENCODING_DICT2[card[1:]]


def vec_to_card(vec):
    # pos=vec.index(1)
    pos = np.where(vec == 1)[0][0]
    return DECODING_DICT1[pos // 13] + DECODING_DICT2[pos % 13]


def pos_to_card(pos):
    return DECODING_DICT1[pos // 13] + DECODING_DICT2[pos % 13]


def card_to_color(card):
    return ENCODING_DICT1[card[0]]


SCORE_DICT = {'SQ': -100, 'DJ': 100, 'C10': 0,
              'H2': 0, 'H3': 0, 'H4': 0, 'H5': -10, 'H6': -10, 'H7': -10, 'H8': -10, 'H9': -10, 'H10': -10,
              'HJ': -20, 'HQ': -30, 'HK': -40, 'HA': -50, 'JP': -60, 'JG': -70}

def a_histoire_relatif(histoire, quel_player):
    res = np.zeros((4, 13, 56))
    for i in range(0, len(histoire), 2):
        res[(histoire[i] - quel_player + 4) % 4][i // 8][4 + histoire[i + 1]] = 1
    for i in range(len(histoire) // 8):
        for j in range(4):
            if i * 8 + j * 2 < len(histoire):
                res[(histoire[i * 8 + j * 2] - quel_player + 4) % 4][i][j] = 1
    return res

def a_standard_input(histoire, quel_player, initial_cards):
    if len(initial_cards) < 20:
        initial_cards = initial_to_formatted(initial_cards)
    his = a_histoire_relatif(histoire, quel_player)
    input = np.zeros((1,1,53,56))
    for i in range(13):
        for j in range(4):
            #print('i,j=',i,',',j)
            input[0,0,4*i+j,:] = his[j][i]
    input[0,0,52,4:] = initial_cards
    return torch.tensor(input)

def a_standard_input_v(histoire, quel_player, initial_cards):
    if len(initial_cards) < 20:
        initial_cards = initial_to_formatted(initial_cards)
    his = a_histoire_relatif(histoire, quel_player).flatten()
    input = np.concatenate((his, initial_cards))
    return torch.tensor(input)


def initial_to_formatted(initialcards):
    res = np.zeros(52)
    for i in initialcards:
        res[card_to_vecpos(i)] = 1
    return res


def trouver_les_choix_feasibles(initial_vec_r, played_vec_r, color_of_this_turn):
    if len(initial_vec_r) > 20:
        initial_vec = initial_vec_r
    else:
        initial_vec = initial_to_formatted(initial_vec_r)
    if len(played_vec_r) > 20:
        played_vec = played_vec_r
    else:
        played_vec = initial_to_formatted(played_vec_r)
    # state is already a 1-dim vector
    # returns a np 01 array
    whats_left = initial_vec - played_vec
    empty_color = False
    if color_of_this_turn == 'A':
        empty_color = True
    elif whats_left[card_to_vecpos(color_of_this_turn + '2'):(card_to_vecpos(color_of_this_turn + 'A')+1)].sum() < 1:
        empty_color = True
    if empty_color:
        return whats_left

    pos = np.where(whats_left == 1)[0]
    pos = pos[pos >= card_to_vecpos(color_of_this_turn + '2')]
    pos = pos[pos <= card_to_vecpos(color_of_this_turn + 'A')]

    res = np.zeros(52)
    for i in range(len(pos)):
        res[pos[i]] = 1
    return res

def create_buffer(buffer, train_data = True):
    #buffer = Memory(53248)
    game = fullgame()
    if train_data:
        f=open('for_shi_batch1.txt')
        parse_for_shi(f, game, buffer)
        f.close()
        torch.save(buffer,'traindata.txt')
    else:
        f = open('for_shi_batch3.txt')
        parse_for_shi(f, game, buffer)
        f.close()
        torch.save(buffer, 'testdata.txt')


if __name__=="__main__":
    try:
        buffer = Memory(53248)
        game = fullgame()
        f=open('for_shi_batch1.txt')
        parse_for_shi(f, game, buffer)
        f.close()
        torch.save(buffer,'traindata.txt')
    except Exception as e:
        print(e)
    finally:
        f.close()
