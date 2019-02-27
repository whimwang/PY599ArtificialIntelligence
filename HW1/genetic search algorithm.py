
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random

#initial
population_size=25;
population = (np.random.rand(population_size,3)-0.5)*8#w1,w2,b
new_population = np.zeros((population_size,3))

tournament_size=5                                          
select_potantial_parents=np.zeros((tournament_size,2)) 
max_generation_num=15                            
mutation_frac=0.4                              
mutation_scale=0.05 

Data_set_size=200
X, Y = make_blobs(n_samples=Data_set_size, centers=2, n_features=2,cluster_std=1.0, center_box=(-4.0, 4.0),random_state=1)
# X is coordinates, Y is class
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=Y))
colors = {0:'red', 1:'blue'}

def perceptron(x1,x2,w1,w2,b):
                     #this is a two input perceptron
                     # we really really should implement neural networks in vector form, otherwise our programs would be awfully slow! 
                     # we had an entire exercise on this. Here just for the sake of simplicity and better understanding of what is happening we are using looping
                     #otherwise vectorization is a must.
  if(0<(x1*w1+x2*w2+b)):
    return(1)
  else:
    return(0)

def error_rate(w1,w2,b):
    whether_same=[perceptron(X[i,0],X[i,1],w1,w2,b) for i in range(X.shape[0])]-Y # same is 0 different is 1
    return(np.mean(np.absolute(whether_same)))
  
def crossover(a,b):              # this function implements the corssover operation, it recives parents a and b, and produces the child c! 
  c=np.random.rand(3)
  beta=np.random.rand(1)
  c[0]=beta*a[0]+(1-beta)*b[0]
  beta=np.random.rand(1)
  c[1]=beta*a[1]+(1-beta)*b[1]
  beta=np.random.rand(1)
  c[2]=beta*a[2]+(1-beta)*b[2]
  return(c)


def mutation(new_population):    # This function implements mutation. It recives the new generation, and mutates mutation_frac of them by adding gaussian noise to them.
  num_of_mutation=math.ceil(len(new_population)*mutation_frac)
  mutation_index=np.random.choice(len(new_population),num_of_mutation, replace=False, p=None)
  new_population[mutation_index,:]=new_population[mutation_index,:]+np.random.normal(0,mutation_scale,(num_of_mutation,3))    
  return(new_population)

def classline(w1,w2,b):
    def f(x1):
       return((-b-w1*x1)/w2)
    return f


for i in range(max_generation_num):
    for j in range(population_size):
       select_potantial_parents=population[np.random.choice(len(population), size=tournament_size, replace=False)] 
       w1_temp = select_potantial_parents[:,0]
       w2_temp = select_potantial_parents[:,1]
       b_temp = select_potantial_parents[:,2]
       #print(min(list(map(error_rate,w1_temp,w2_temp,b_temp))))#check error rate
       parent_1 = select_potantial_parents[np.argmin(list(map(error_rate,w1_temp,w2_temp,b_temp))),:]

       select_potantial_parents=population[np.random.choice(len(population), size=tournament_size, replace=False)] 
       w1_temp = select_potantial_parents[:,0]
       w2_temp = select_potantial_parents[:,1]
       b_temp = select_potantial_parents[:,2]
       #print(min(list(map(error_rate,w1_temp,w2_temp,b_temp))))#check error rate
       parent_2 = select_potantial_parents[np.argmin(list(map(error_rate,w1_temp,w2_temp,b_temp))),:]

       new_population[j,:]=crossover(parent_1,parent_2)

    new_population = mutation(new_population)
   
   
 
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    #plot classification line
    for j in range(0,population_size):
         f = classline(new_population[j,0],new_population[j,1],new_population[j,2])
         t = np.arange(-6.5, 2.5, 0.2)
         plt.plot(t, f(t), 'y--')
    plt.show() 
    
    # plot w1,w2,b
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(new_population[:,0],new_population[:,1],new_population[:,2])
    ax.set_xlabel('w1 Label')
    ax.set_ylabel('w2 Label')
    ax.set_zlabel('b Label')
    
    population=new_population.copy()  
    

#print(population)
