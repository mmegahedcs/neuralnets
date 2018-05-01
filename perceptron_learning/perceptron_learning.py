import numpy as np

def sgnWX(num):
    if num >= 1:    
        return 1
    elif num <= -1:
        return -1

def updateWeight(eta,weight,pattern,cls,sgn):
    return weight + eta * (cls-sgn) * pattern

class_1 = [[0,1],[-1,1],[-1,-1],[-1,0]]
class_2 = [[0,-1],[1,-1],[1,1],[1,0]]
initial_weight =np.array([1,0,0])
print("Initial Weight: ",initial_weight)
bias = 1
for c1 in class_1:
    c1.insert(0,bias)
for c2 in class_2:
    c2.insert(0,bias)
classes = {'1':class_1,'-1':class_2}

def perceptron_learn (classes, initial_weight):
    done = True
    for cls in classes:
        for ptrn in classes.get(cls):
            prtn_array=np.array([ptrn])
            WX = (prtn_array * initial_weight).sum()
            
            if int(cls) >= 1: y=min(int(cls),WX)
            elif int(cls) <= -1: y=max(int(cls),WX)
           
            if y != int(cls):
                initial_weight = updateWeight(1, initial_weight, prtn_array, int(cls), sgnWX(WX))
                print ("Updated Weight: ", initial_weight)
                done=False

    return initial_weight,done

initial_weight,done= perceptron_learn (classes, initial_weight)


done=False
while done==False:
    initial_weight,done= perceptron_learn (classes, initial_weight)

        