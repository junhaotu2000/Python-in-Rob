import copy
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

#### Return value and variables ####
# print(*obejcts, sep = ' ', end = '\n', file = sys.stdout, flush=False)
# parameters -- def a funtion // arguments -- use a function
# *obejcts - any inputs are acceptable
name = input("What's your name? ")

# remove whitespace from str
name  = name.strip() 

# capitalize user's name (first word)
name = name.capitalize()

# capitalize user's name (all words)
name = name.title()

# replace a word
name = name.replace("_", 'do it')

# split the first name and last name 
first, last = name.split(" ")

# f-strings (format string)
print(f"Hello world! {name}") 

# the most standard way
print("Hello again!",name) 

# treat all as a string 
print("Hello, "+ name) 

# add a identifier to end the print 
print("Hello, ", end = "") 
print(name)

# sep the output with a sepcial sign
print("Hello, ", name, sep = "???") 

# print the quoet inside the sentence 
print('Hello, "friend"') 
print("Hello, \"friend\"")

# multiple line input - in sperate line
my_str = f"""I
am
a
Geek !"""
print(my_str) 

# multiple line input - in single line
my_str = f"""I \
am \
a \
Geek !"""
print(my_str)

