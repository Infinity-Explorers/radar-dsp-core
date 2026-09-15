import numpy as np
import time

#  بجرب حاجة بس 
signals= np.array([1.0,2.0,3.0,4.0])
amplified= signals*2.5
print(amplified)

# creating 5 points equel from 1 to 10
x = np.linspace(0,10,5)
print (x)


#creating nums from 0 to 10 with 2 steps
y= np.arange(0,10,2)
print(y)

#create array of num
angles = np.array([0,np.pi/2,np.pi])
sin_values = np.sin(angles)
clean_sin_values = np.round(sin_values)
cos_values = np.cos(angles)
clean_cos_values = np.round(cos_values)
result = sin_values + 5
print("sin:",sin_values)
print("cos",cos_values)
print(result)
print(clean_sin_values)
print(clean_cos_values)

#comparison
N =1000000
#using for loop
t_start = time.time() # start time 
x_list = np.linspace(0,10,N)
sin_list = []
for val in x_list:
    sin_list.append(np.sin(val))
t_loop = time.time()- t_start
print(f"the time of the loop:{t_loop:.4f} second") 

#using numpy vectorization
t_start = time.time()
x_vec = np.linspace(0,10,N)
sin_vec = np.sin(x_vec)
t_vec = time.time() - t_start
print(f"the time of the vectorization:{t_vec:.4f} second") 




