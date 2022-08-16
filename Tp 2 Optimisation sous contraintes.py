#!/usr/bin/env python
# coding: utf-8

# # Partie 1 : Discrétisation et résolution directe

# ## partie théorique 
# 

# *- Pour trouver une solution approchée de ce problème de minimisation, on procède comme on
# a fait dans le TP 1, on discrétise l'intervalle  [0 , 1] en n + 1 sous intervalles égaux [x i , x i+1 ] ,
# i = 0, . . . , n , où x i = ih avec h = 1/(n + 1) . Puis, on approche la solution u(x) par une fonction
# uh (x) , continue, affine par morceaux et donnée par*:
#   
#   
#   $U h$$($x$)$ $=$ $\sum_{i=1}^n u_i $$\phi_i$($x$)
#   
#   
#   avec :
#   
#   
#   
# $$
# \phi_i(x) = \left\{
#     \begin{array}{ll}
#         \frac{x_i - x_{i-1}}{h} \ {x} \in [x_{i−1} , x_i ], \\
#         \frac{x_{i+1} - x}{h} \ {x} \in [x_i , x_{i+1} ], \\
#          \ 0 & \mbox{sinon.}
#     \end{array}
# \right.
# $$
# 

# In[8]:


import numpy as np
from scipy.optimize import minimize


# ##  Implémentation des fonctionnelles Jn et ∇Jn

# In[11]:


#Implémenter les fonctionnelles à travers les deux fonctions Python suivantes :
# J(U, x, fx) et DJ(U, x, fx)

#''' Definir les fonctions f , g , J et DJ '''


def j(u,x,fx):
    
    n=u.size
    
    h=x[1]-x[0]
    
    A=np.zeros((n,n))
    for i in range(n):
        for j in range(n): 
            if i==j:
                A[i,i]=2/h
                if j-1 >=0:
                    A[i,j-1]=-1/h
                if j+1 <=n-1:
                    A[i,j+1]=-1/h
                    
    
   
    b=h*fx
    r=0.5*np.vdot(np.dot(A,u),u)-np.vdot(b,u)
    
    return(r)
                  
    
def deltaJ(u,x,fx):
    n=u.size
    
    h=x[1]-x[0]
    
    A=np.zeros((n,n))
    for i in range(n):
        for j in range(n): 
            if i==j:
                A[i,i]=2/h
                if j-1 >=0:
                    A[i,j-1]=-1/h
                if j+1 <=n-1:
                    A[i,j+1]=-1/h
                    
    
   
    b=h*fx
    
    return np.dot(A,u)-b


# ## Résolution du  notre problème pour f(x) = 1

# In[12]:


# 1èr Test pour f(x) =1

def f1(x):
    return  1

def g1(x):
      
        a=1.5 - 20*(x - 0.6)**2
        
        if a<0:
            return  0
        else:
            return a
        
f=np.vectorize(f1)
g=np.vectorize(g1)
        
n = 100
x = np . linspace (0 ,1 , n + 2 )

xv = x [ 1 : - 1 ]
fv , gv = f ( xv ) , g ( xv )

Jf = lambda u : j(u,xv,fv)

DJf = lambda u : deltaJ(u,xv,fv)

const = ( { 'type': 'ineq' , 'fun' : lambda u : u - gv ,
'jac' : lambda u : np.eye (np.size(u))})

u = np.zeros (n)

res = minimize ( Jf , u , method = 'SLSQP' , jac = DJf , constraints = const ,
tol = 1e-8 , options = { 'xtol': 1e-8 , 'disp': True , 'maxiter': 5000 } )

s1=res.x
s1


# ##  Résolution du  notre problème pour  $f $($x$)  $= $ $\pi^2$ $\sin $($\pi x$)
#  

# In[13]:


# 2ème Test pour f(x) =   𝜋^2sin(𝜋 x)

def f1(x):
    return  np.pi**2*np.sin(np.pi*x)

def g1(x):
      
        a=1.5 - 20*(x - 0.6)**2
        
        if a<0:
            return  0
        else:
            return a
        
f=np.vectorize(f1)
g=np.vectorize(g1)
        
n = 100
x = np . linspace (0 ,1 , n + 2 )

xv = x [ 1 : - 1 ]
fv , gv = f ( xv ) , g ( xv )

Jf = lambda u : j(u,xv,fv)

DJf = lambda u : deltaJ(u,xv,fv)

const = ( { 'type': 'ineq' , 'fun' : lambda u : u - gv ,
'jac' : lambda u : np.eye (np.size(u))})

u = np.zeros (n)

res = minimize ( Jf , u , method = 'SLSQP' , jac = DJf , constraints = const ,
tol = 1e-8 , options = { 'xtol': 1e-8 , 'disp': True , 'maxiter': 5000 } )

s2=res.x
s2


# ##   Vérification des conditions pour la solution (1)

# In[14]:


print("U(0) = ", round(s1[0]))
print("U(1) = ", round(s1[99]))


# In[15]:


s1>gv


#  *<p style="color:red">on remarque donc que U(x) > g(x),
# donc la solution numérique vérifie la condition 4.b</p>*
# 

# In[16]:



# calcule de U"(x)
n=100
# préparation des tableaux qui vont recevoir les valeurs

xnew = np.zeros(n-1)
yp = np.zeros(n-1)

# calcul des abscisses et des valeurs de la dérivée 1
for i in range(n-1): 
    xnew[i] = (x[i] + x[i+1]) / 2
    yp[i] = (s1[i+1] - s1[i]) / (x[i+1] - x[i])

# préparation des tableaux qui vont recevoir les valeurs
    
xnew1 = np.zeros(n-2)
yp1 = np.zeros(n-2)

# calcul des abscisses et des valeurs de la dérivée 2
for i in range(n-2): 
    xnew1[i] = (xnew[i] + xnew[i+1]) / 2
    yp1[i] = (yp[i+1] - yp[i]) / (xnew[i+1] - xnew[i])

# vérificatio que u" (x) ≥ f (x)

yp1 >= fv[:98]


# <p style="color:red">la solution numérique ne vérife pas la condition   4.a</p>*

# In[17]:


(yp1 - fv[0:98])* (s1[0:98] - gv[0:98]) ==0


# <p style="color:red"> donc la solution numérique ne vérifie pas  la condition 4.c </p>

# ##    Augmentation de la nombre  ***n*** :

# __on faisant augmenter le nombre n de 100 vers 200 on ramarque que quelque condition sont vérifées :__
# 
# __<p style="color:red">__
#     U(0) = U(1) =0 </p>    
# 
# __<p style="color:red">__
#     U(x) > g(x) </p>**

# # Partie 2 : Méthode du gradient projeté

# ## Ecrire un programme projK.py qui prend en argument un point u et g n = (g(x i )) i=1,...,n et qui renvoie $P_k(u)$

# In[110]:


def projection(v,g) :
    Pk=np.maximum(v, g )
    return Pk


# ## Implémentation 
# 

# In[18]:


#Méthode du gradient projeté à pas fixe  

def gradient_projete_fixe (J , DJ , gn , u0 , rho , Tol , iterMax , store ) :
    
    k=0
    r=Tol
    u=u0
    l=[]
    while (k < iterMax and r >= Tol ):
        w=-DJ(u)
        Pk=np.maximum(u+rho*w, gn )
        v=u
        u=Pk
        r=np.linalg.norm(v-u)
        k+=1
        l.append(u)
    if (store==1):
        return l
    else:
        return l[-1]
      
    
        


# ## Pour n = 2 , tracer sur une même figure les courbes de niveaux de J 2 ainsi que le champ de vecteurs $∇J_2$ sur le pavé [−10, 10] × [−10; 10] , calculer les itérations $u (k)$ = (u 1 , u 2 ) données par l'algorithme de gradient projeté à pas fixe, et tracer sur la même figure que précédemment la ligne qui relie les $u(k)$ . On prendra $u(0) = (8, 4)$ , $ρ = 0.1 et Tol = 10 −

# In[28]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

v1 = np.linspace(-10,10,50)
v2 = np.linspace(-10,10,50)

v,u = np.meshgrid(v1,v2)

z = v*(2*v-u) + u*(2*u-v) -u -v


#Pour le courbe en 3D:
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(v,u,z)

#Pour le contour
plt.contour(v,u,z)
plt.show()

plt.quiver(v,u,v2,v1,Z,scale=1500,units='width')

plt.show()


# In[36]:


n = 2
    

x = np . linspace (0 ,1 , n + 2 )
xv = x [ 1 : - 1 ]
fv , gv = f ( xv ) , g ( xv )
Jf = lambda u : j(u,xv,fv)

DJf = lambda u : deltaJ(u,xv,fv)
u0=np.array(n*[4])
    
l1= gradient_projete_fixe(Jf, DJf,gv,u0,0.1,10^-5,100,1)

xx=[i[0] for i in l1]
yy=[i[1] for i in l1]

plt.plot(xx,yy)


# ## Pour n = 5, 20, 50, 100 , affcher à l'aide de la fonction print le nombre d'itérations ainsi que le temps de calcul pour chaque n . Tracer sur une même figure les solutions approchées $U_n$ , ainsi que le graphe de la fonction $g$ . On prendra $ρ$ = 0.1 , Tol = 10 −5

# In[41]:


import time
from datetime import timedelta

l=[5,20,50,100]
for i in l:
    start_time = time.monotonic()
 
    n = i
    
    x = np . linspace (0 ,1 , n + 2 )

    xv = x [ 1 : - 1 ]
    fv , gv = f(xv) , g(xv)
    Jf = lambda u : j(u,xv,fv)

    DJf = lambda u : deltaJ(u,xv,fv)
    u0=np.array(n*[4])
    
    l1= gradient_projete_fixe(Jf, DJf,gv,u0,0.1,10^-5,200000,1)
    
    end_time = time.monotonic()
    
    print("le temps de calcul pour n = "+str(i)+"\n",timedelta(seconds=end_time - start_time))


    print("le nombre d'itération pour n = "+str(i)+"\n",len(l1))
    


# In[65]:


#Représentation de la solution u et la fonction g dans la mm graphe 

UU=l1[-2]
xv =np.linspace (0 ,1 , n + 2 )[1:-1]

plt.plot(xv,UU)
plt.plot(xv,gv)
plt.show()


# ## Reprendre l'expérience précédente pour ρ = 0.5 , puis ρ = 1 . Que constate-t-on ? Peut-on choisir le pas ρ arbitrairement ?

# In[66]:


#pour ρ = 0.5

import time
from datetime import timedelta

l=[5,20,50,100]
for i in l:
    start_time = time.monotonic()
 
    n = i
    
    x = np . linspace (0 ,1 , n + 2 )

    xv = x [ 1 : - 1 ]
    fv , gv = f(xv) , g(xv)
    Jf = lambda u : j(u,xv,fv)

    DJf = lambda u : deltaJ(u,xv,fv)
    u0=np.array(n*[4])
    
    l1= gradient_projete_fixe(Jf, DJf,gv,u0,0.5,10^-5,200000,1)
    
    end_time = time.monotonic()
    
    print("le temps de calcul pour n = "+str(i)+"\n",timedelta(seconds=end_time - start_time))


    print("le nombre d'itération pour n = "+str(i)+"\n",len(l1))
    


# In[89]:


#pour ρ = 1

import time
from datetime import timedelta

l=[5,20,50,100]
for i in l:
    start_time = time.monotonic()
 
    n = i
    
    x = np . linspace (0 ,1 , n + 2 )

    xv = x [ 1 : - 1 ]
    fv , gv = f(xv) , g(xv)
    Jf = lambda u : j(u,xv,fv)

    DJf = lambda u : deltaJ(u,xv,fv)
    u0=np.array(n*[4])
    
    l1= gradient_projete_fixe(Jf, DJf,gv,u0,1,10^-5,200000,1)
    
    end_time = time.monotonic()
    
    print("le temps de calcul pour n = "+str(i)+"\n",timedelta(seconds=end_time - start_time))


    print("le nombre d'itération pour n = "+str(i)+"\n",len(l1))
    


# ***On constate que ,pour ρ =0.5 le temps de calcule ainsi que le nombre d'itération sot dimuniées 
# aussi pou  ρ =1 , ces deux dernier ont diminuées aussi .***
# 

# ## Reprendre les questions 2.5 et 2.6 avec ρ optimal :

# In[109]:


#Calcule du ρ optimale 
n=2

h=x[1]-x[0]
    
A=np.zeros((n,n))
for i in range(n):
    
     for j in range(n): 
            
            if i==j:
                A[i,i]=2/h
                if j-1 >=0:
                    A[i,j-1]=-1/h
                if j+1 <=n-1:
                    A[i,j+1]=-1/h
                    
l = np.linalg.eigvals(A)
l=np.sort(l)

rho=2/(l[0]+l[-1])
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

v1 = np.linspace(-10,10,50)
v2 = np.linspace(-10,10,50)

v,u = np.meshgrid(v1,v2)

z = v*(2*v-u) + u*(2*u-v) -u -v


#Pour le courbe en 3D:
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(v,u,z)

#Pour le contour
plt.contour(v,u,z)
plt.show()

plt.quiver(v,u,v2,v1,Z,scale=1500,units='width')

plt.show()

#######

# tracer sur la même figure que précédemment la ligne qui relie les u(k)

n = 2
    

x = np . linspace (0 ,1 , n + 2 )
xv = x [ 1 : - 1 ]
fv , gv = f ( xv ) , g ( xv )
Jf = lambda u : j(u,xv,fv)

DJf = lambda u : deltaJ(u,xv,fv)
u0=np.array(n*[4])
    
l1= gradient_projete_fixe(Jf, DJf,gv,u0,rho,10^-5,100,1)

xx=[i[0] for i in l1]
yy=[i[1] for i in l1]

plt.plot(xx,yy)


# In[113]:


#pour ρ = ρ_optimal 
import numpy as np
import time
from datetime import timedelta

l=[5,20,50,100]
for i in l:
    start_time = time.monotonic()

    n = i
    
    x = np . linspace (0 ,1 , n + 2 )
    
    
    
    h=1/n
    
    A=np.zeros((n,n))
    
    for k in range(n):
    
       for j in range(n): 
            
            if k==j:
                A[k,k]= 2/h
                if j-1 >=0:
                    A[k,j-1]= -1/h
                if j+1 <=n-1:
                    A[k,j+1]= -1/h
     
   
    l= np.linalg.eigvals(A)
    l=np.sort(l)
    
    rho=2/(l[0]+l[-1])
     
     
 
    xv = x [ 1 : - 1 ]
    fv , gv = f(xv) , g(xv)
    Jf = lambda u : j(u,xv,fv)

    DJf = lambda u : deltaJ(u,xv,fv)
    u0=np.array(n*[4])
    l1= gradient_projete_fixe(Jf, DJf,gv,u0,rho,10^-5,10000,1)
    
    end_time = time.monotonic()
    
    print("le temps de calcul pour n = "+str(i)+"\n",timedelta(seconds=end_time - start_time))


    print("le nombre d'itération pour n = "+str(i)+"\n",len(l1))
    


# **On constate que pour n=5 par exemple 
# :**
# 
# |  la valeur de $ρ$  |   le temps de calcule   |   Nombre d'itération  | 
# |---    |:-:    |:-:    |
# | 0.1                | 0:00:01.199728          |18355                  |           
# | 0.5                |    0:00:00.029258       |   432                 | 
# |   1.0              |    0:00:00.006494       |  79                   |    
# | $ρ$ optimale       |   0:00:00.055997  |  599 |  
# 
# 

# ## Reprendre la question précédente pour f (x) = π 2 sin(πx)

# In[112]:


# 2ème Test pour f(x) =   𝜋^2sin(𝜋 x)

def f1(x):
    return  np.pi**2*np.sin(np.pi*x)

def g1(x):
      
        a=1.5 - 20*(x - 0.6)**2
        
        if a<0:
            return  0
        else:
            return a
        
f=np.vectorize(f1)
g=np.vectorize(g1)
        
n = 100
x = np . linspace (0 ,1 , n + 2 )

xv = x [ 1 : - 1 ]
fv , gv = f ( xv ) , g ( xv )

Jf = lambda u : j(u,xv,fv)

DJf = lambda u : deltaJ(u,xv,fv)


#pour ρ = ρ_optimal 

import numpy as np
import time
from datetime import timedelta

l=[5,20,50,100]
for i in l:
    start_time = time.monotonic()

    n = i
    
    x = np . linspace (0 ,1 , n + 2 )
    
    
    
    h=1/n
    
    A=np.zeros((n,n))
    
    for k in range(n):
    
       for j in range(n): 
            
            if k==j:
                A[k,k]= 2/h
                if j-1 >=0:
                    A[k,j-1]= -1/h
                if j+1 <=n-1:
                    A[k,j+1]= -1/h
     
   
    l= np.linalg.eigvals(A)
    l=np.sort(l)
    
    rho=2/(l[0]+l[-1])
     
    
 
    xv = x [ 1 : - 1 ]
    fv , gv = f(xv) , g(xv)
    Jf = lambda u : j(u,xv,fv)

    DJf = lambda u : deltaJ(u,xv,fv)
    u0=np.array(n*[4])
    l1= gradient_projete_fixe(Jf, DJf,gv,u0,rho,10^-5,10000,1)
    
    end_time = time.monotonic()
    
    print("le temps de calcul pour n = "+str(i)+"\n",timedelta(seconds=end_time - start_time))


    print("le nombre d'itération pour n = "+str(i)+"\n",len(l1))
    

