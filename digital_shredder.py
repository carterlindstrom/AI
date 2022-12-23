import sys
import math

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z']
    
    for letter in alphabet:
        X[letter] = 0
        
        
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        letters = f.read()
        
        for letter in letters:
            if letter.isalpha():
                X[letter.upper()] += 1
        
    f.close()

    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!


def main():
    
    q1 = shred("letter.txt")
    print("Q1")
    for value in q1:
        print(value + " " + str(q1[value])) 
    
    param_vectors = get_parameter_vectors()
    
    print("Q2")
    x1 = q1['A']
    e1 = param_vectors[0][0]
    s1 = param_vectors[1][0]
    print(f'{x1 * math.log(e1):0.4f}')
    print(f'{x1 * math.log(s1):0.4f}')
    
    print("Q3")
    
    eng_sums = 0
    counter = 0
    for value in q1:
        pi = param_vectors[0][counter]
        eng_sums += math.log(pi) * q1[value] 
        counter += 1
    eng = math.log(0.6) + eng_sums
    print(f'{eng:0.4f}')
    
    spa_sums = 0
    counter = 0
    for value in q1:
        pi = param_vectors[1][counter]
        spa_sums += math.log(pi) * q1[value] 
        counter += 1
    spa = math.log(0.4) + spa_sums

    print(f'{spa:0.4f}')
    
    
    print("Q4")
    final_p = 0
    if spa - eng >= 100:
        final_p = 0
    elif spa - eng <= -100:
        final_p = 1
    else:
        final_p = 1 / (1 + math.exp(spa - eng))
        
    print(f'{final_p:0.4f}')
    
    
main()