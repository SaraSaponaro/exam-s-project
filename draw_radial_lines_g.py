'''
definisco la funzione che fa disegnare i raggi.
In particolare l'output è una maschera binaria (della stessa dimensione dell'immagine input), dove NL indica il numero di raggi di lunghezza R che partono dal centro della lesione interessata.
'''

import numpy as np


'definisco le funzioni che mi permettono il passaggio da coordinate cartesiane a polari'
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return(x, y)



def draw_radial_lines(ROI,center,R,NL):

    'creo i miei angoli come suddivisione in NL parti dello angolo giro'
    theta=np.linspace(0,2*np.pi,NL)

    'creo una lista vuota'
    Ray_masks=[]

    'looppo su ogni theta'
    for _ in range(1,NL):

        'creo un vettore contenente il valore dei miri raggi per un dato theta'
        rho=np.arange(1,R)

        'passo dalle coordinate polari a quelle cartesiane'
        x,y= pol2cart(theta[_],rho)

        'centro la origine delle linee nel centro della lesione che ho dato in imput (center_x, center_y)'
        iir=center_x+round(x)
        jjr=center_y+round(y)


        'creo una tabella cioè vettori messi in verticale'
        line1=np.column_stack((iir,jjr))

        ' metto i valori della roi a zero per ?? '
        for __ in range(1,len(line1[0])):
            ROI(line1[__,1],line1[__,2])=0

        'ho creato una matrice (futura maschera) di zeri'
        Ray_mask=np.zeros((len(ROI[0]),len(ROI[1])))

        'e adesso faccio lo stesso procedimento di prima solo che opra ho la dimensione giusta'
        for __ in range(1,len(line1[0])):
            Ray_mask(line1[__,1],line1[__,2])=0


        'in teoria ora devo fare il cat, coiè concatenare in 3 dimensioni le matrici Ray_masks,Ray_mask'



