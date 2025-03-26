import math
import random
from pathlib import Path
from PIL import Image
from collections import OrderedDict
import pickle

take_ready_weights = True
continue_traning = False

path_to_ready_network = "weights_knife_16x16_v2_fourth.dat"
path_to_save_new_network = "weights_knife_16x16_v2_fourth.dat"


def normalize(digit_to_norm, mind = 0, maxd = 255):
    norm = (digit_to_norm - mind) / (maxd - mind)
    return norm

def get_train_set_7():
    directory1 = 'Knifes_v2'
    pathlist = Path(directory1).glob('*.png')
    trainset = {}
    for path in pathlist:
        print(path)
        img = Image.open(path)
        obj = img.load()
        x = 0
        y = 0
        currentpic = ''
        while x < 16:
            y = 0
            while y < 16:
                curp = img.getpixel((x, y))
                norm1 = normalize(curp[0])
                norm2 = normalize(curp[1])
                norm3 = normalize(curp[2])
                currentpic += str(norm1) + '_' + str(norm2) + '_' + str(norm3) + '_'
                y += 1
            x += 1
        currentpic = currentpic[:len(currentpic)-1]
        #print(currentpic)
        trainset.update({currentpic:1})
    return trainset

def get_train_set_not_7():
    directory1 = 'Not knifes_v2'
    pathlist = Path(directory1).glob('*.png')
    trainset = {}
    for path in pathlist:
        print(path)
        img = Image.open(path)
        obj = img.load()
        x = 0
        y = 0
        currentpic = ''
        while x < 16:
            y = 0
            while y < 16:
                curp = img.getpixel((x, y))
                norm1 = normalize(curp[0])
                norm2 = normalize(curp[1])
                norm3 = normalize(curp[2])
                currentpic += str(norm1) + '_' + str(norm2) + '_' + str(norm3) + '_'
                y += 1
            x += 1
        currentpic = currentpic[:len(currentpic)-1]
        #print(currentpic)
        trainset.update({currentpic:0})
    return trainset

def get_pic_pixels():
    directory1 = 'Test knife'
    pathlist = Path(directory1).glob('*.png')
    trainset = {}
    for path in pathlist:
        img = Image.open(path)
        obj = img.load()
        x = 0
        y = 0
        currentpic = ''
        while x < 16:
            y = 0
            while y < 16:
                #print(img.getpixel((x, y))[0])
                curp = img.getpixel((x, y))
                norm1 = normalize(curp[0])
                norm2 = normalize(curp[1])
                norm3 = normalize(curp[2])
                currentpic += str(norm1)+'_'+str(norm2)+'_'+str(norm3)+'_'

                y += 1
            x += 1
        currentpic = currentpic[:len(currentpic)-1]
        return currentpic

def get_knife_pixels(img):
        obj = img.load()
        x = 0
        y = 0
        currentpic = ''
        while x < 16:
            y = 0
            while y < 16:
                #print(img.getpixel((x, y))[0])
                curp = img.getpixel((x, y))
                norm1 = normalize(curp[0])
                norm2 = normalize(curp[1])
                norm3 = normalize(curp[2])
                currentpic += str(norm1)+'_'+str(norm2)+'_'+str(norm3)+'_'

                y += 1
            x += 1
        currentpic = currentpic[:len(currentpic)-1]
        return currentpic


#input()

def activation_sigmoid(x):
    return 1/(1+math.e**-x)

def error_mse(list_i, list_a):
    answer = 0
    for i in range(len(list_i)):
        answer += (list_i[i]-list_a[i])**2
    answer = answer/len(list_i)
    return answer


#print(activation(0.78))
#print(error_mse([1], [0.33]))

#input_neurons ={'i_1':1, 'i_2':0}
input_neurons = {}
i1 = get_pic_pixels().split('_')
i = 1
for ids in i1:
    input_neurons.update({'i_'+str(i):float(ids)})
    i += 1

#layer_neurons = {  # layer_neurons = h1, h2 в примере сайта
#    'inputs':{'h_1':0, 'h_2':0},
#    'outputs':{'h_1':0, 'h_2':0}
#                 }

layer_o = {
    'inputs':{'o_1':0},
    'outputs':{'o_1':0}
    }

#weights = {
#    'i_1?h_1':{'value':0.45, 'last_change':0},
#    'i_2?h_1':{'value':-0.12, 'last_change':0},
#    'i_1?h_2':{'value':0.78, 'last_change':0},
#    'i_2?h_2':{'value':0.13, 'last_change':0},
#    'h_1?o_1':{'value':1.5, 'last_change':0},
#    'h_2?o_1':{'value':-2.3, 'last_change':0}
#    }

def generate_weights(input_neurons_amount, layer_neurons_amount, output_neurons_amount):
    weights = {}
    for i in range(input_neurons_amount):
        for h in range(layer_neurons_amount):
            weights.update({'i'+'_'+str(i+1)+'?'+'h'+'_'+str(h+1):{'value':random.randint(-400, 400)/100, 'last_change':0}})

    for h in range(layer_neurons_amount):
        for o in range(output_neurons_amount):
            weights.update({'h'+'_'+str(h+1)+'?'+'o'+'_'+str(o+1):{'value':random.randint(-400, 400)/100, 'last_change':0}})

    if take_ready_weights:
        with open(path_to_ready_network, 'rb') as dump_in:
            weights = pickle.load(dump_in)
    return weights

def generate_layer_neurons(amount):
    layer_neurons = {
        'inputs':{},
        'outputs':{}
        }
    for h in range(amount):
        layer_neurons['inputs'].update({'h'+'_'+str(h+1):0})
        layer_neurons['outputs'].update({'h'+'_'+str(h+1):0})
    return layer_neurons


def shuffle_dict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    return OrderedDict([(k, d[k]) for k in keys])

#train_set = {'0_0':0, '0_1':1, '1_0':1, '1_1':0}

train_set = {}
for key, value in get_train_set_7().items():
    train_set[key] = value

for key, value in get_train_set_not_7().items():
    train_set[key] = value

train_set = shuffle_dict(train_set)



def get_o_ideal(input_neurons, train_set):
    string = ''
    for idss in input_neurons:
        string += str(input_neurons[idss])+'_'
    string = string[:len(string)-1]
    o_ideal = train_set[string]
    return o_ideal

def get_fin(x):
    return (1-x)*x

def set_neural_coefficients(input_neurons, layer_neurons, layer_o, weights):
    for ids in layer_neurons['inputs']:
        layer_neurons['inputs'][ids] = 0
        for idss in input_neurons:
            weight = weights[idss+"?"+ids]['value']
            layer_neurons['inputs'][ids] += input_neurons[idss]*weight
            
    for ids in layer_neurons['outputs']:
        layer_neurons['outputs'][ids] = activation_sigmoid(layer_neurons['inputs'][ids])

    for ids in layer_o['inputs']:
        layer_o['inputs'][ids] = 0
        for idss in layer_neurons['outputs']:
            weight = weights[idss+'?'+ids]['value']
            layer_o['inputs'][ids] += layer_neurons['outputs'][idss]*weight
            
    for ids in layer_o['outputs']:
        layer_o['outputs'][ids] = activation_sigmoid(layer_o['inputs'][ids])

    return layer_neurons, layer_o

def calc_answer(input_neurons, layer_neurons, layer_o, weights):
    list_a = []
    layer_neurons, layer_o = set_neural_coefficients(input_neurons, layer_neurons, layer_o, weights)
    for ids in layer_o['outputs']:
        list_a.append(layer_o['outputs'][ids])
    return list_a[0]

def calc_error(input_neurons, layer_neurons, layer_o, weights, train_set):
    layer_neurons, layer_o = set_neural_coefficients(input_neurons, layer_neurons, layer_o, weights)
    
    string = ''
    for idss in input_neurons:
        string += str(input_neurons[idss])+'_'
    string = string[:len(string)-1]

    o_ideal = train_set[string]
    #print(o_ideal)
    #input()

    list_a = []
    for ids in layer_o['outputs']:
        list_a.append(layer_o['outputs'][ids])

    
    list_i = [o_ideal]

    #print(list_a)
    
    error_final = error_mse(list_i, list_a)
    #print(error_final)
    #input()
    return error_final, layer_neurons, layer_o

def error_backpropagation(input_neurons, layer_neurons, layer_o, weights, train_set, E=0.2, a=0.4):
    for ids in layer_o['outputs']:
        o_actual = layer_o['outputs'][ids]
        fin_o = get_fin(o_actual)
        o_ideal = get_o_ideal(input_neurons, train_set)
        delta_o = (o_ideal-o_actual) * fin_o
        o_weights = []
        for idss in weights:
            if idss.split('?')[1] == ids:
                o_weights.append(idss)
        for idss in o_weights:
            #print(idss.split('?')[0]) 
            h_out = layer_neurons['outputs'][idss.split('?')[0]]
            current_w = weights[idss]
            #print(current_w)
            #input()
            current_w_value = current_w['value']
            fin_h = get_fin(h_out)
            delta_h = fin_h*(current_w_value*delta_o)

            gradw = h_out*delta_o
            #print(gradw)
            #input()

            last_delta_w = current_w['last_change']
            delta_w = E*gradw + a*last_delta_w
            #print(delta_w)
            #input()
            w = current_w['value'] + delta_w
            current_w['value'] = w
            current_w['last_change'] = delta_w
            #print(current_w['value'])
            #input()
   
            h_weights = []
            for idsss in weights:
                #print(idsss, " ", idss)
                if idss.split('?')[0] == idsss.split('?')[1]:
                    h_weights.append(idsss)
            #print(h_weights)
            #input()
            for idsss in h_weights:
                i_out = input_neurons[idsss.split('?')[0]]
                #print("i_out=", i_out)
                current_w = weights[idsss]
                current_w_value = current_w['value']
                #print("current_w_value=", current_w_value)
                last_delta_w = current_w['last_change']
                #print("last_delta_w=", last_delta_w)
                gradw = i_out*delta_h
                #print("gradw=", gradw)
                delta_w = E*gradw + a*last_delta_w
                #print("last_delta_w=", last_delta_w)
                w = current_w_value + delta_w
                #print("w=", w)
                #input()
                weights[idsss]['value'] = w
                weights[idsss]['last_change'] = delta_w
                
                
    return weights


layer_neurons_amount = 20
layer_neurons = generate_layer_neurons(layer_neurons_amount)
weights = generate_weights(input_neurons_amount=768, layer_neurons_amount=layer_neurons_amount, output_neurons_amount=1)

layer_neurons, layer_o = set_neural_coefficients(input_neurons, layer_neurons, layer_o, weights)          
#print(layer_neurons)
#print(layer_o)
#input()

def teach_cycle(input_neurons, layer_neurons, layer_o, weights, train_set):
    stop_teaching = False
    for ids in train_set:
        ii = ids.split('_')
        input_neurons = {}
        icount = 1
        for idss in ii:
            input_neurons.update({'i_'+str(icount):float(idss)})
            icount += 1
        #i_1 = int(ids.split('_')[0])
        #i_2 = int(ids.split('_')[1])
        #input_neurons = {'i_1':i_1, 'i_2':i_2}
        
        
        error_final, layer_neurons, layer_o = calc_error(input_neurons, layer_neurons, layer_o, weights, train_set)
        print("Ошибка "+str(i+1)+": ",error_final*100, "%")
        #print(error_final)
        
        weights = error_backpropagation(input_neurons, layer_neurons, layer_o, weights, train_set, E=0.1, a=0.8)
        #print("Веса скорректированы.")
        if error_final == 0.0:
            stop_teaching = True
            break
    return layer_neurons, layer_o, weights, stop_teaching, error_final

if not take_ready_weights or continue_traning:
    while True:
        for i in range(10):
            train_set = shuffle_dict(train_set)
            layer_neurons, layer_o, weights, stop_teaching, error = teach_cycle(input_neurons, layer_neurons, layer_o, weights, train_set)
            if stop_teaching:
                break
            if i == 10:
                print("!!!!!!!!!!!!!!!!!!!!Ошибка: ",error*100, '%')

        with open(path_to_save_new_network, 'wb') as dump_out:
            pickle.dump(weights, dump_out)

    
#print("Ответ: ",calc_answer(input_neurons = {'i_1':0, 'i_2':1}, layer_neurons = layer_neurons, layer_o = layer_o, weights = weights))

def getpicanswer(img):
    i1 = get_knife_pixels(img).split('_')
    inputs = {}
    i = 1
    for ids in i1:
        inputs.update({'i_' + str(i): float(ids)})
        i += 1
    ans = calc_answer(input_neurons=inputs, layer_neurons=layer_neurons, layer_o=layer_o, weights=weights)
    # print("Ответ: ",round(calc_answer(input_neurons = inputs, layer_neurons = layer_neurons, layer_o = layer_o, weights = weights), 4))
    #if ans >= 0.7:
    #    textans = "Этот нож занесён."
    #elif ans <= 0.3:
    #    textans = "Этот нож НЕ занесён."
    #else:
    #    textans = "Я не знаю"
    #print("Ответ: " + textans + ' (числовой результат: ' + str(round(ans, 4)) + ')')
    if ans >= 0.90:
        return True
    else:
        return False

while True:
    break
    i1 = get_pic_pixels().split('_')
    inputs = {}
    i = 1
    for ids in i1:
        inputs.update({'i_'+str(i):float(ids)})
        i += 1
    ans = calc_answer(input_neurons = inputs, layer_neurons = layer_neurons, layer_o = layer_o, weights = weights)
    #print("Ответ: ",round(calc_answer(input_neurons = inputs, layer_neurons = layer_neurons, layer_o = layer_o, weights = weights), 4))
    if ans >= 0.7:
        textans = "Этот нож занесён."
    elif ans <= 0.3:
        textans = "Этот нож НЕ занесён."
    else:
        textans = "Я не знаю"
    print("Ответ: "+textans+' (числовой результат: '+str(round(ans, 4))+')')
    
    input()
#input_neurons={'i_1':i1, 'i_2':i2}





            
