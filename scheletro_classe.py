import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
import os
import argparse
import logging
from scipy.optimize import curve_fit                                                     
logging.basicConfig(level=logging.INFO)    

class Rivelatore_cilindrico:
    def __init__(self, name, radius, distance, lunghezza):
        self.name=name
        self._radius=radius
        self._distance=distance
        self._lunghezza=lunghezza

    def print_info(self):

        print('Il rivelatore "{}" ha raggio pari a {} cm , è lungo {} cm e dista dalla sorgente {} cm'.format(self.name,self.radius,self.lunghezza,self.distance))

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if(value<0):
            print("Il rivelatore non può avere raggio negativo")
            print('Il raggio sarà settato a 0.5 cm')
            self._radius=0.5
        else:
            self._radius=value

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        if(value<0):
            print('per ora va bene ma poi dovrai cambiare gli angoli')
        else:
            self.distance=value

    @property
    def lunghezza(self):
        return self._lunghezza

    @radius.setter
    def lunghezza(self, value):
        if(value<0):
            print("Il rivelatore non può avere lunghezza negativa")
            print('la lunghezza ssarà settata a 2 cm')
            self._lunghezza=2
        else:
            self._lunghezza=value


class Sorgente:
    def __init__(self, name, activity, emilife, energy1, energy2):
        self.name=name
        self.activity=activity
        self.emilife=emilife
        self.energy1=energy1
        self.energy2=energy2

    def print_info(self):
        if(self.energy1 == 0 or self.energy2 == 0):
            print('la sorgente di {} ha una attività di {} KBq, una emivita di {} ed emette con energia di {} Kev'.format(self.name,self.activity,self.emilife,self.energy1+self.energy2) )
        else:
            print('la sorgente di {} ha una attività di {} KBq, una emivita di {} ed emette con energia di {} Kev e di {} Kev'.format(self.name,self.activity,self.emilife,self.energy1,self.energy2))

    

    def N_particles(self, time):
        tau=self.emilife/(np.log(2))
        N=-self.activity*tau*(1-np.exp(time/tau))
        print(N)
        return N


def efficienza_intrinseca(m,n,d,r,l):     #esiste di sicuro una cosa piu efficiente


    if m!= 1:
        x=[]

        for i in range (0,m):
            t0=time.time()

            phi=np.random.uniform(0,np.pi,size=n)       #provo a generarli solo sulla parte frontale-> phi tra 0 e pi e theta rimene cosi

            t=np.random.random(size=n)
            theta=np.arccos(1-2*t)

            #faccio lo smearing sulle misure di lunghezza


            R = r + 0.005 * (1-2*random.random())
            L = l + 0.1 * (1-2*random.random())
            D = d + 0.005 * (1-2*random.random())
            theta_max=np.arctan((R)/(L+D))
            phi_max=np.arctan((R)/(L+D))

            theta_zero=np.zeros(n)
            phi_zero=np.zeros(n)

            prova_theta=np.where(theta >= (np.pi)/2 - theta_max , theta, theta_zero )
            theta_pass=np.where(theta <= (np.pi)/2 + theta_max , prova_theta, theta_zero )
            prova_phi=np.where(phi >= (np.pi)/2 - phi_max , phi, phi_zero )
            phi_pass=np.where(phi <= (np.pi)/2 + phi_max , prova_phi, phi_zero )

            
            vagone_finale=theta_pass+phi_pass
            mask_fin=vagone_finale>2*((np.pi)/2 - theta_max)
            x.append(len(vagone_finale[mask_fin])/(2*n))
            #x.append(len(vagone_finale[mask_fin])/(n))

            #print(len(vagone_finale[mask_fin])/n)
            print (m-i)

        x=np.asarray(x)

        plt.figure('histo_accettanza m = {}'.format(m))
    
        ydata, edges, _ = plt.hist(x, bins=20, density=True)
        xdata = 0.5 * (edges[1:] + edges[:-1])


        def gaussian(x, a, mu, sigma):
            return a/(sigma*np.sqrt(2*np.pi))*(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.))))

        popt, pcov = curve_fit(gaussian, xdata, ydata, p0=[300000,0.0003, 0.00001])
        print(' i parametri trovati sono (a, mu, sigma)',popt)
        print(' le relative incertezze sono di ',np.sqrt(pcov.diagonal()))
        print(1/popt[1])
        _x = np.linspace(0, 0.0005, 500)
        _y = gaussian(_x, *popt)
        #plt.figure('gauss_fit')
        plt.plot(_x, _y)

        mask = ydata > 0
        chi2 = sum(((ydata[mask] - gaussian(xdata[mask], *popt)) / np.sqrt(ydata[mask]))**2.)
        nu = mask.sum() - 3
        sigma = np.sqrt(2 * nu)
        print('chi2, dof, incetezza chi2',chi2, nu, sigma)
    
    else:
        #print (len(vagone_finale[mask_fin])/(2*n))
        #print (len(vagone_finale[mask_fin])/(n))
        #print('l accettanza stimata per il rivelatore {} è di {} '.format(NaI.name,len(vagone_finale[mask_fin])/n))
        #print('stimata in un tempo di {} s'.format(time.time()-t0))
        #plt.hist(phi, bins=200)
        #plt.hist(theta, bins=200)
        #return(len(vagone_finale[mask_fin])/n)
        return(len(vagone_finale[mask_fin])/(2*n))
        



        

if __name__ == '__main__':
    NaI = Rivelatore_cilindrico('NaI',0.25,7.945,1.67)
    #Cs137=Sorgente('Cesio_137', 53.9, 952092792, 662, 0)
    accettanza=efficienza_intrinseca(10**4,10**6,NaI.distance,NaI.radius,NaI.lunghezza)

    plt.show()


    







    
