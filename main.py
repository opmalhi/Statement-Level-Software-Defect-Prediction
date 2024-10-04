import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype 
from tensorflow import keras 
from keras.layers import *
from keras.models import *
from keras.activations import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import os
import itertools
import string
import tensorflow as tf
from itertools import product
import math
from utils import lex
from utils import yacc
from utils import cpp
import zipfile

zip_name = "./newdata.zip"

# The collumns containing the code info
cols = ["code", "block"]
# The Archive Containing The Actual Codes
archive = zipfile.ZipFile('newdata.zip', 'r')
# The first name is the name of the containing folder of the codes 
list_files = archive.namelist()[1:]
# Now We will make a scanner for the C++ language
scanner = lex.lex(cpp)
# If any thing was considered fault at the line i, we will consider all the lines [i - range_n, i + range_n) to be fault 
range_n = 4
# Then We Define The literals of the program
lits = cpp.literals
# Then We Define The Tokens
toks = list(cpp.tokens)
# We remove the White Space token to add it later 
toks.remove("CPP_WS")
# We add the White Space token here because we want it to have the value of zero, we'll use this latter for padding lines of code
toks.insert(0, "CPP_WS")
# Tok 2 N : a dictionary from tokens to thier integer, mapped, value
tok2n = dict(zip(toks + [i for i in lits], itertools.count()))
# N 2 Tok : a dictionary from integers to thier token, mapped, value
n2tok = dict(zip(itertools.count(), toks + [i for i in lits]))

# The maximum value we allow in as a constant value in a code
max_v = 2147483647 - 1

# The amount of importance we give to 1s 0s and false postives and false negatives
WEIGHTS_FOR_LOSS = np.array([[2,0.5],[0.1,0.1], [0.1,0.1], [0.1,0.1], [0.1,0.1]]) #TODO: add same false positive and negative as error line to other classes 
#added temporary weights for classes 2,3,4

def get_loss_function(weights, rnn=True):
        
    '''
    gives us the loss function
    '''
    def w_categorical_crossentropy_mine(y_true, y_pred):
        nb_cl = len(weights)
        
        if(not rnn):
            final_mask = K.zeros_like(y_pred[:, 0])
            y_pred_max = K.max(y_pred, axis=1)
            y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += ( weights[c_t, c_p] * K.cast(y_pred_max_mat, tf.float32)[:, c_p] * K.cast(y_true, tf.float32)[:, c_t]  )
            return K.categorical_crossentropy(y_true, y_pred, True) * final_mask 
        else:
            final_mask = K.zeros_like(y_pred[:, :,0])
            y_pred_max = K.max(y_pred, axis=2)
            y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], K.shape(y_pred)[1], 1))
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += ( weights[c_t, c_p] * K.cast(y_pred_max_mat, tf.float32)[:, :,c_p] * K.cast(y_true, tf.float32)[:, :,c_t]  )
            return K.categorical_crossentropy(y_true, y_pred, True) * final_mask 

            
    return w_categorical_crossentropy_mine

import sys
def get_data(list_files, archive, log = False):
    '''
    reads the data and handles the range_n number
    '''
    res = []
    for i in list_files:
        try :
            x = pd.read_csv(archive.open(i), sep = "`")
            x = x[x.columns[:-1]]
            res.append(x)
        except Exception: 
            print(i)
            continue
    resF = []
    for n_i, i in enumerate(res) :
        
        if i.shape[0] == 0 :
            continue
            
        a = i.values
        b = i.copy()
#         This line of code will change the data from (Features, class) to (Features, one-hot-vector)
        out_classes = [0, 1, 2, 3, 4]

       
        cat_type = CategoricalDtype(categories=out_classes, ordered=True)
        try:
            s_cat = i.iloc[:, -1].astype(int)
        except:
            print(f'i have shape {i.shape} but i choose to misbehave')
            continue
       
        s_cat = s_cat.astype(cat_type)
        one_hot_encodings = pd.get_dummies(s_cat, dtype=int)
        # print(f'type: {type(one_hot_encodings)}\n shape: {one_hot_encodings.shape}')
        
        b = b.drop(b.columns[-1], axis=1) 
        # print(f'num cols after label drop: {b.shape}\n num OHE: {one_hot_encodings.shape} ')
        # Concatenate the original DataFrame with the dummy variables DataFrame
        b = pd.concat([b, one_hot_encodings], axis=1) #uses pd.get_dummies to one-hot encode instead of exclusive or
        b = b.values
        # print(f'encoding shape: {b.shape}')
        # b = np.concatenate([b[:, :-1], b[:, -1:].astype(int) ^ 1, b[:, -1:]], axis = -1) #TODO: change this so the last cols represent all classes
        
        
        for j in range(len(b)):
            if np.sum(a[j - range_n : j + range_n, -1]) > 0 :
#                 This was explained before the declaration of range_n
#               TODO: decide strategy to represent all lines around errors if multiple errors are encountered in close proximity 
#               TODO: maybe give first preference to the first occuring error
                # b[j, -1] = 1
                # b[j, -2] = 0

                #steps:
                #1. find first non-zero element
                # Find the first non-zero index within the range
                # first_nonzero = a.iloc[j - range_n : j + range_n, -1][a.iloc[j - range_n : j + range_n, -1] != 0].idxmin()
                first_nonzero = next((num for num in a[j - range_n : j + range_n, -1] if num != 0), None)

                #assign label based on non-zero element encountered
                if first_nonzero == 1:
                    b[j, -1] = 0
                    b[j, -2] = 0
                    b[j, -3] = 0
                    b[j, -4] = 1
                    b[j, -5] = 0
                elif first_nonzero == 2:
                    b[j, -1] = 0
                    b[j, -2] = 0
                    b[j, -3] = 1
                    b[j, -4] = 0
                    b[j, -5] = 0
                elif first_nonzero == 3:
                    b[j, -1] = 0
                    b[j, -2] = 1
                    b[j, -3] = 0
                    b[j, -4] = 0
                    b[j, -5] = 0
                else:              #if first_nonzero_index == 4:
                    b[j, -1] = 1
                    b[j, -2] = 0
                    b[j, -3] = 0
                    b[j, -4] = 0
                    b[j, -5] = 0
                


        for x in range(len(b)):
            for y in range(len(b[x])):
#                 Here we will try to change any thing that is not the code it self and which is a string into numbers 
                if y > 1 :
                    if type(b[x, y]) == str :
                        try :
                            float(b[x, y].strip())
                        except Exception : 
                            b[x, y] = -3
                elif y == 1 :
                    b[x, y] = "DATA DOES NOT MATTER"
#         By 0s we mean the code being fine and so on
        b = pd.DataFrame(b, columns=list(i.columns)[:-1] + ["0s", "1s", "2s", "3s", "4s"]) #changed num classes, added 2,3,4
        b.replace("#empty", np.nan, inplace =True)
        resF.append(b.dropna())
        
    if log :     
        print("data was read and changed")    
    return resF
        
        
    
    
def get_replacement(scanner, string_in):
    
    '''
    gets a string and returns the None, 2 which is the tokenized version
    '''
    try :
        scanner.input(string_in)
    except Exception as e :
        print("Exception in using the lex", e)
        print(string_in)
    token = scanner.token()
    
    
#     id2n and n2id are the same as n2tok tok2n but they are extended to contain the information of the symbol table of each code separately
    id2n = dict(zip([i for i in lits], [tok2n[i] for i in lits]))
    n2id = dict(zip([tok2n[i] for i in lits], [i for i in lits]))
    
    n_id = len(lits) + 1
    
    
    res = []
    
    while token is not None :
        
        t = token.type
        
#         If we have recieved a token and it is not something we need to use ord for
        if t in cpp.tokens :
#             Reciving a white space
            if token.type == cpp.tokens[cpp.tokens.index("CPP_WS")]:
                #this is because this will make it easier for us to pad our data
                v = 0
#             Reciving an ID from the code
            elif token.type == cpp.tokens[cpp.tokens.index("CPP_ID")]:
                v = token.value
#                 Checking if need to add the id to n2id or not
                if v in id2n.keys() :
                    pass
                else :
                    id2n[v] = n_id
                    n2id[n_id] = v
                    
                    n_id += 1
                v = id2n[v]
#             If we receive a string (We don't use the value of strings)
            elif token.type == cpp.tokens[cpp.tokens.index("CPP_STRING")]:
                v = -1
#             If we recive #
            elif token.type == cpp.tokens[cpp.tokens.index("CPP_POUND")]:
                v = -2
#             If we recive ##
            elif token.type == cpp.tokens[cpp.tokens.index("CPP_DPOUND")]:
                v = -3
#             If we recive char
            elif token.type == cpp.tokens[cpp.tokens.index("CPP_CHAR")]:
                v = -4
            elif token.type in cpp.tokens[3:]:
                print("some thing went really wrong")
#             Parsing the value of constant values
            else:
                try :
                    tv = token.value.lower()
                    if tv[-1] == "l" : 
                        tv = tv[:-1]
                    if tv[-1] == "u" : 
                        tv = tv[:-1]
                    if "x" in  tv :
                        v = int(tv, base = 16)
                    elif tv[-1].lower() == "l":
                        if tv[-2].lower() == "u" :
                            v = float(tv[:-2])
                        else :
                            v = float(tv[:-1])
                    else :
                        v = float(tv)
                    v = np.clip(v, - max_v, max_v)
                    
                except Exception as e :
                    print("Couldn't scan this number", token)
                    return
                
                
            
            
        else :
            v = ord(t)
        try :
            t = tok2n[t]
        except Exception :
            n = len(id2n.keys()) + 1 
            tok2n[t] = n
            n2tok[n] = t
            id2n[t] = n
            n2id[n] = t
            t = tok2n[t]
            
        res.append([t, v])
        token = scanner.token()
        
    res = np.array(res)
    
    return res
        
    
    
def tokenize_data(data):
    
    '''
    reads data and tokenizes each of the sentences and adds them together.
    The out put will contain the actual data, max number of lines per code and mean number of lines per code
    the actual data will have the following shape :
    
    Number of codes, Number of lines per each code , 2 (Data and State)
    State will contain (Code being right, Code Being Wrong)
    Data Will Contain (Number Of Words, 2 (Token, Value))
    
    '''
    
    res = []
    x = []
    mean = 0
    max_num = 0
    for i in data:
#         If We had any code submissions that was empty, we skip them
        if i.shape[0] == 0 :
            continue 
        temp = []
        mean += i.shape[0]
        max_num = max(max_num, i.shape[0])
        
        for j in i.values :
            
            try :
                tok = get_replacement(scanner, j[0]).astype(np.float32)
            except Exception as e :
                continue
                
            x.append(tok)
            
            #TODO change j[-2:] so it gets all available classes 
            y = j[-5:] #changed range to 5 classes instead of 2
            temp.append([tok, y])
            
        res.append(temp)
    mean /= len(res)
    
    return res, mean, max_num
            
    
def change_cols(num, res, empty):
    
    '''
    
    pads or removes data so they all have the same shape in one code  file 
    
    num : amount of word we'll have per each line
    empty : what we'll use to pad our data with
    
    '''
    
    resF = []
    temp_i = 0
    for i in res :
        
        temp = []
        
        if (len(i) == 0):
#           We'll any coding file which is empty
            continue 
            
        for j in i :
            
#             J[0] is the data and J[1] is the state

            if len(j[0]) < num :
                
                result = np.concatenate([j[0], np.ones(( num - len(j[0]), 2)) * empty], axis = 0)
                
            elif len(j[0]) > num :
                result = j[0][:num, :]
            else :
                result = j[0]
                
            result = result.reshape((-1))
            
#             This is so that we'll have the data and our state at the same time
            result = np.concatenate([result, np.array([j[1]]).reshape((-1))], axis = 0)
            temp.append(np.array(result))

        # print(f'temp shape at {temp_i}: {len(temp)}')
        # temp_i += 1
        resF.append(np.array(temp))
        
       
    resF = np.asarray(resF, dtype="object")
    
    
    return resF

def get_final_data(tokenized_final, data):
    
    
    '''
    adds the information from the parser to the things that were gained from the information of scanners
    tokenized_final will be the output of "change_cols" and data will be the output of "get_data"
    '''
#     The first line reads data and drops the following columns : columns containing text of the parser or lex and the 
#     The last two columns which are the state of the code which we are trying to predict
    dataR = np.concatenate([i.drop(cols, axis = 1).values[:, :-5] for i in data], axis = 0) #TODO change -2 so it matches len of available labels
    dataR = dataR.astype(np.float32)
    
    cnt = 0 
    
    res = []
    
    
    for i in tokenized_final : 
        temp = []
        for j in i :
            
            add = dataR[cnt, :]
            temp.append(np.concatenate([add, j], axis = 0))
            
            cnt += 1
            
            
        res.append(np.array(temp))
    res = np.asarray(res, dtype="object")
    return res

def gather_data(list_data, archive, scaler = None, add_all = False, type_add = 0,
                pad1 = None, pad2 = None, return_before_pad = False, cons_per_line = 10, log  = False):
    '''
    Reading, Tokenizing, Concatenating And Normalizing Data
    list_data : Name Of The Codes We Are Using
    scalar : Scalar Used To Normalize Data, If None Is Presented, The Function Will Compute One
    add_all : Whether Or Not We Want All Our Codes To Have The Same Amount As For The Lines Of Code
    type_add : The amount of lines each code should contain : {0 : mean, 1 : max number of lines}
    pad1 : The amount of words each line should contain, If None is presented const_per_line + mean(amount of words per line) would be used
    pad2 : The amount of lines each code should contain, type_add would not be used if pad2 is not None
    return_before_pad : Whether or not to also return the data before it was padded to have the same amount of lines percode 
    '''
#     First We Read Our Data From The Zip File
    data = get_data(list_data, archive)
#     The We tokenize our data
    r, mean, max_num = tokenize_data(data)
    
#     Then We Create Our Empty Vector
    empty = np.array([tok2n["CPP_WS"], 0]).reshape(1, 2).astype(np.float32)
    
#     The Defualt Option for Padding 
    if pad1 is None :
        pad1 = int(mean) + cons_per_line
        
#     We Padd Our Data At Each Line With Extra White Spaces
    res = change_cols(pad1, r, empty)
    r = np.array(res)
    
#     Here we will concatenate our lexical features and our preprocessed features
    r = get_final_data(r, data)
    if log :
        print("Padded The Lexical And Preprocessed Features of Data")
    
    
    res = np.asarray(r).astype('object')
    
    if add_all :
        
        if log :
            print("Computing How Many Empty Line To Add To Codes ")
        if pad2 is None :
            
            mean = 0
            max_num = -1
            for i in r :
                mean += i.shape[0]
                max_num = max(max_num, i.shape[0])
                
                
            mean /= r.shape[0]
            nums = [int(mean), max_num]
            pad2 = nums[type_add]
            
        res = []
        if log :
            print("Computed How Many Empty Line To Add To Codes ")
        for i in r :
            
            if i.shape[0] < pad2 :
                
                zeros = np.zeros([pad2 - i.shape[0], i.shape[1]])
                zeros[:, -2] = 1 #TODO change -2 so it matches len of available labels #changed to -5
                temp = np.concatenate([i, zeros], axis = 0)
            elif i.shape[0] > pad2 :
                temp = i[:pad2, :]
            else :
                temp = i
            res.append(temp)
        if log :
            print("Added All The Empty Lines ")

    res = np.array(res)
    
    save_r = r.copy()
    
    r = np.concatenate(res, axis = 0).astype(np.float32)
    
    
    if scaler is None :
        
        scaler = StandardScaler().fit(r[:, :-5].astype(np.float32)) #TODO change -2 so it matches len of available labels #changed to -5
        if log :
            print("Computed Mean And Standard Deviation For Normalization ")
    
    for i, iv in enumerate(res) :
        
        res[i, :, :-5] = scaler.transform(iv[:, :-5].astype(np.float32)).astype(np.float32) #TODO change -2 so it matches len of available labels #changed to -5
        
        if log :
            print("Data Was Normalized ")

    
    if return_before_pad :
        return res, scaler, pad1, pad2, save_r
        
    return res, scaler, pad1, pad2
            
            
                
                
                 
                
    
    
def get_model(shape):
    
    
    '''
    gets the first rnn model
    shape : shape of the input : shape of the codes [Number of lines (which can be None), Number of Fetures per line]
    '''
    in1 = Input(shape)
    X = Bidirectional(LSTM(150, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(in1)
    X = LSTM(150, return_sequences=True, dropout=0.25, recurrent_dropout=0.1,)(X)
    X = Dropout(0.2)(X)
    X = Dense(256, activation=relu)(X)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = Dense(128, activation=relu)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.25)(X)
    X = Dense(64, activation=relu)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = Dense(32, activation=relu)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.4)(X)
    X = Dense(16, activation=relu)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.4)(X)
    
    X = Dense(5, activation=softmax)(X) #TODO change 2 so it matches len of available labels # changed out layer to 5
    
    
    
    
    model = Model(in1, X)
    
    
    
    return model
    
def get_acc(y_true, y_pred):
    
    # acc = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]) #TODO change labels to match the num of classes #added 2,3,4
    acc = accuracy_score(y_true, y_pred)
    rec1 = recall_score(y_true, y_pred, average='weighted')
    prec1 = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    #replacement formulae
    #TP = sum(acc[i][i])
    #FN = sum(acc[i]) - acc[i][i]. ie sum across rows minus TP
    #FP = sum(acc[][i]) - acc[i][i]. ie. sum across column - TP
    #TN = sum(acc) - (sum(acc[i]) + sum(acc[][i])) 

    #calculate per class, then use weighted avg since specific error classified as other column shouldn't be as bad 
#     True Positive
#     tp =  acc[1][1]
# #     False Negative
#     fn =  acc[1][0]
# #     False Positive
#     fp =  acc[0][1]
# #     True Negative
#     tn =  acc[0][0]
# #     Recall
#     rec1 = acc[1][1] / (acc[1][1] + acc[1][0])
# #     Precision 
#     prec1 = acc[1][1] / (acc[1][1] + acc[0][1])
# #     Over ALl Accuracy
#     accuracy = (acc[1][1] + acc[0][0]) / (acc[1][1] + acc[0][0] + acc[1][0] + acc[0][1])
# #     F1 Accuracy
#     f1 = 2.0 / ((1.0/rec1) + (1.0/prec1))
    
    return rec1, prec1, acc, f1 #, tp , fn , fp , tn,  accuracy changed to acc
    
def get_mus(y_true, x, model):
    
    
    y_t = np.argmax(y_true, axis = -1).reshape((-1))
    y_p = model.predict(x)
    y_p = np.argmax(y_p, axis=-1).reshape((-1))
    
    
    
    return get_acc(y_t, y_p)
    
    
    # K is for our K-fold
k = 4
# The name of the codes we want to use, you can slice this list to a smaller list for a fast test
l = list_files[:]
# The number of codes we use on each fold
size = math.ceil(len(l) / k)
# Verbose for our NN model 
verbose = 0 

# The results for trains and tests respectivly
trs = []
ts = []

for i in range(k):
    
    print("k", i)
#     start and end will be the indicies for what we'll use for test
    start  = i * size
    end    = min(len(l), (i + 1) * size)
    
    data_train = l[:start] + l[end:]
    data_test  = l[start : end]
    
    if len(data_test) <= 0 or len(data_train) <= 0 :
        print("hey")
        continue

   
    # gathering data for train
    r_train, scaler, pad1, pad2 = gather_data(data_train, archive, add_all = True)
    # gathering data for test, please note that the same mean and standard deviation that was computed for train will be used to 
    # normalize test data and also the information of pad1 and pad2 is computed from train so that no information will be 
    # leaked from train and also none of the aforementioned are dependent on the test data 
    r_test, _, _, _ = gather_data(data_test, archive, scaler = scaler, add_all = True, pad1 = pad1, pad2 = pad2)
    
    print("data read")
    
    # configuring model
    model = get_model([None, r_train.shape[-1] - 5])
    loss = get_loss_function(WEIGHTS_FOR_LOSS)
    model.compile(tf.keras.optimizers.Adam(learning_rate = 1e-3), keras.losses.categorical_crossentropy, metrics = ["accuracy"])

    
    # making the X, y for train and test set
    X_train = r_train[:, :, :-5] #changed -2 to -6 since there are 5 classes
    y_train = r_train[:, :, -5:] #changed -2 to -6 since there are 5 classes
    
    temp_line = '--' * 50
    print(f'Variable shapes\n{temp_line}\nX_train: {X_train.shape}\n\ny_train: {y_train.shape} ')

    X_test = r_test[:, :, :-5] #changed -2 to -6 since there are 5 classes
    y_test = r_test[:, :, -5:] #changed -2 to -6 since there are 5 classes

    print(f'Variable shapes\n{temp_line}\nX_test: {X_test.shape}\n\ny_test: {y_test.shape} ')

    X_train = tf.convert_to_tensor(X_train, tf.float32)
    X_test = tf.convert_to_tensor(X_test, tf.float32)

    # y_train = np.array(y_train).astype(np.int_)
    # y_test = np.array(y_test).astype(np.int_)

    y_train = tf.convert_to_tensor(y_train, tf.int32)  
    y_test = tf.convert_to_tensor(y_test, tf.int32)

    

    print(f'Params and their types \n X_train: {type(X_train[0])}\ny_train: {type(y_train[0][0])}\nX_test: {type(X_test[0])}\ny_test: {type(y_test[0][0])} ')
    
    # training the model
    print("training on the data started ")
    model.fit(X_train, y_train, validation_data = [X_test, y_test], epochs = 20, batch_size = 8, verbose = verbose)
    print("training on the data finished ")
    
    # saving and printing the accuracy of training data
    print("train : rec1, prec1, accuracy, f1, acc , tp , fn , fp , tn ")
    trs.append(get_mus(y_train, X_train,model))
    print(trs[-1])

    # preparing to write the training accuracy to a file
    strRes = "train : "
    for counter in range(len(trs[-1])):
        strRes = strRes + '%.5f' % trs[-1][counter] + " , "
    
    strRes += " \n "
    

    # saving and printing the accuracy of testing data
    print("test : rec1, prec1, accuracy, f1, acc, tp , fn , fp , tn ")
    ts.append(get_mus(y_test, X_test,model))
    print(ts[-1])
    
    # preparing to write the testing accuracy to a file
    strRes += " test :  "
    for counter in range(len(ts[-1])): #changed 8 to len(ts[-1]) to get real tuple len
        strRes = strRes + '%.5f' % ts[-1][counter] + " , "
    
    # writing the accuracies on to a file
    f = open("test.txt", "a")
    f.write(strRes + "\n")
    f.close()
                        

trs = np.array(trs)
ts = np.array(ts)
print("avg train : rec1, prec1, accuracy, f1, acc")
print(np.mean(trs[:, : 4], axis=0))
print("avg test : rec1, prec1, accuracy, f1, acc")
print(np.mean(ts[:, : 4], axis=0))