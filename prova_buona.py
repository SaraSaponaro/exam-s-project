import pylab as plt
import numpy as np
import math
import imageio
from scipy.signal import convolve2d
from skimage.transform import resize
from scipy.ndimage.filters import generic_filter
from skimage.filters import threshold_otsu
"leggo il file"
#fileID='0016p1_2_1.png'
#fileID='0025p1_4_1.pgm'
fileID='0036p1_1_1.png'
image=imageio.imread(fileID)
'''
plt.figure()
plt.imshow(image)
plt.show()
'''

#%% parametri
smooth_factor= 8
scale_factor= 8
size_nhood_variance=5
NL=33

#%%processo l'img
k = np.ones((smooth_factor,smooth_factor))/smooth_factor**2
im_conv=convolve2d(image, k )
im_resized = resize(im_conv, (126,126), preserve_range=True)
im_norm = im_resized/np.max(im_resized)

plt.figure('immagine normalizzata')
plt.imshow(im_norm)
plt.figure('immagine')
plt.imshow(image)
#plt.show()

#%%SEGMENTATION
"per il momento seleziono una roi a mano dell'immagine"
y1=29
y2=90
x1=30
x2=97

ROI=np.zeros(np.shape(im_norm))
ROI[y1:y2,x1:x2]=im_norm[y1:y2,x1:x2]
x_max, y_max=np.where(ROI==np.max(ROI))

#%%radial lines
R=int(np.sqrt((x2-x1)**2+(y2-y1)**2)/2)     #intero più vicino 
center=[np.min(x_max),np.min(y_max)]
nhood=np.ones((size_nhood_variance,size_nhood_variance))

'definisco le funzioni che mi permettono il passaggio da coordinate cartesiane a polari'
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

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
        iir.append(-center[0]+int(x))
        jjr.append(-center[1]+int(y))

    'creo una tabella cioè vettori messi in verticale'
    line1=np.column_stack((iir,jjr))

    'ho creato una matrice (futura maschera) di zeri'
    Ray_mask=np.zeros(np.shape(ROI))

    for ___ in range(0,len(line1)):
        i=line1[___][0]
        j=line1[___][1]
        Ray_mask[i,j]=1 
    Ray_masks.append(Ray_mask)
    
plt.figure('raggio casuale')
plt.imshow(Ray_masks[10])
plt.imshow(im_norm, alpha=0.5)
plt.plot(center[0],center[1], 'r.')
#plt.show()

#%% max variance points
'''funsione che binarizza img'''
def imbinarize(img):
    thresh = threshold_otsu(img)
    img[img >= thresh] = 1
    img[img < thresh] = 0
    return img

J = generic_filter(im_norm, np.std, size=size_nhood_variance)
'''
plt. figure()
plt.imshow(J, cmap='gray')
ptl.show()
'''

B_points=[]
roughborder=np.zeros(np.shape(im_norm))
p_x=[]
p_y=[]
d=[]
for _ in range (0, NL):
    Jmasked=J*Ray_masks[_]     #J*raggi=maschera dell'img
    Jmasked=Jmasked*imbinarize(ROI)
    w = np.where(Jmasked==np.max(Jmasked))
    p_y.append(w[0][0])     
    p_x.append(w[1][0])
    d.append(Jmasked[w[0][0],w[1][0]])
    roughborder[p_x, p_y]=im_norm[w[0][0], w[1][0]]

plt.figure('bordo puntallato')
plt.imshow(roughborder)
plt.imshow(im_norm, alpha=0.3)
plt.plot(p_x,p_y)   #dovrebbe vwnire piu o meno cosi
#plt.show()


#%% cerchiamo di unire i pixel per fare il bordo
'mi serve dopo-> scrivo una fuzione per trovare la distanza euclidea'
def distanza(x1,y1,x2,y2):
    distanza_euclidea=np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distanza_euclidea


'questa liste mi rappresentano le coordinate x e y dei punti che mi formeranno il bordo'
bordo_x=[]
bordo_y=[]


'inizio il tutto partendo da un punto casuale del bordo: alla fine dovrà coincidere con ultimo per poter chiudere il bordo'

a=0
b=0
plt.figure()
for _ in range(0,1):
    start = im_norm[p_x[_], p_y[_]]         #pixel da cui parto
    stop = im_norm[p_x[_+1], p_y[_+1]]      #pixel in cui finisco

    'pur essendo poco ottimizzato, io per ogni pixel ho 8 vicini, salvo le loro coordinate in una lista'

    N1=im_norm[p_x[_]-1, p_y[_]]
    N2=im_norm[p_x[_]-1, p_y[_]+1]
    N3=im_norm[p_x[_], p_y[_]+1]
    N4=im_norm[p_x[_]+1, p_y[_]+1]
    N5=im_norm[p_x[_]+1, p_y[_]]
    N6=im_norm[p_x[_]+1, p_y[_]-1]
    N7=im_norm[p_x[_]+1, p_y[_]-1]
    N8=im_norm[p_x[_]-1, p_y[_]-1]

    vicino_x=[p_x[_]-1,p_x[_]-1,p_x[_],p_x[_]+1,p_x[_]+1,p_x[_]+1,p_x[_]+1,p_x[_]-1]
    vicino_y=[p_y[_],p_y[_]+1,p_y[_]+1,p_y[_]+1,p_y[_],p_y[_]-1,p_y[_]-1,p_y[_]-1]

    '''plt.scatter(p_x[_], p_y[_],label='start')
    plt.scatter(p_x[_]-1, p_y[_],label='vicino')
    plt.scatter(p_x[_]-1, p_y[_]+1,label='vicino')
    plt.scatter(p_x[_], p_y[_]+1,label='vicino')
    plt.scatter(p_x[_]+1, p_y[_]+1,label='vicino')
    plt.scatter(p_x[_]+1, p_y[_],label='vicino')
    plt.scatter(p_x[_]+1, p_y[_]-1,label='vicino')
    plt.scatter(p_x[_]+1, p_y[_]-1,label='vicino')
    plt.scatter(p_x[_]-1, p_y[_]-1,label='vicino')'''

    

    'ora devo caloclare la distanza euclidde tra il vicino e stop'
    distanza_list=[]
    c=0
    for __ in range(0,len(vicino_x)):
        c=distanza(vicino_x[__],vicino_y[__],p_x[_+1], p_y[_+1])
        distanza_list.append(c)

    distanza_list=np.asarray(distanza_list)
    d=np.where(distanza_list==distanza_list.min())

    bordo_x.append(vicino_x[int(d[0])])
    bordo_y.append(vicino_y[int(d[0])])
    
  
    


#plt.legend()
#plt.show()


