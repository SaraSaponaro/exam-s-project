import numpy as np
import math

def draw_radial_lines(ROI,center,R,NL):
    'definisco le funzioni che mi permettono il passaggio da coordinate cartesiane a polari'
    def pol2cart(rho, theta):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return(x, y)

    'creo i miei angoli come suddivisione in NL parti dello angolo giro'
    theta=np.linspace(0,2*np.pi,NL)

    'creo una matrice vuota'
    Ray_masks=[]

    'creo un vettore contenente il valore dei miei raggi per un dato theta'
    rho=np.arange(R)

    for _ in range(0,NL):
        iir=[]
        jjr=[]
        for __ in range (0, R):
            'passo dalle coordinate polari a quelle cartesiane'
            x,y = pol2cart(rho[__],theta[_])
            'centro la origine delle linee nel centro della lesione che ho dato in imput (center_x, center_y)'
            iir.append(center[0]+math.ceil(x))
            jjr.append(center[1]+math.ceil(y))

        'creo una tabella cioè vettori messi in verticale'
        line1=np.column_stack((iir,jjr))

        'ho creato una matrice (futura maschera) di zeri'
        Ray_mask=np.zeros(np.shape(ROI))

        for ___ in range(0,len(line1)):
            i=line1[___][0]
            j=line1[___][1]
            Ray_mask[j,i]=1
        Ray_masks.append(Ray_mask)

    return Ray_masks
