import pylab as plt
import numpy as np
import math
import imageio
from scipy.signal import convolve2d
from skimage.transform import resize
from scipy.ndimage.filters import generic_filter
from skimage.filters import threshold_otsu
"leggo il file"
fileID='0016p1_2_1.png'
#fileID='0025p1_4_1.pgm'
#fileID='0036p1_1_1.pgm'
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
NL=32

#%%processo l'img
k = np.ones((smooth_factor,smooth_factor))/smooth_factor**2
im_conv=convolve2d(image, k )
im_resized = resize(im_conv, (126,126), preserve_range=True)
im_norm = im_resized/np.max(im_resized)

plt.figure()
plt.imshow(im_norm)
plt.show()

#%%SEGMENTATION
"per il momento seleziono una roi a mano dell'immagine"
y1=47
y2=82
x1=43
x2=80

ROI=np.zeros(np.shape(im_norm))
ROI[y1:y2,x1:x2]=im_norm[y1:y2,x1:x2]
x_max, y_max=np.where(ROI==np.max(ROI))

#%%radial lines
R=math.ceil(np.sqrt((x2-x1)**2+(y2-y1)**2)/2)
center=[x_max[0],y_max[0]]
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
Ray_masks=np.zeros(np.shape(ROI))


'creo un vettore contenente il valore dei miei raggi per un dato theta'
rho=np.arange(R)

for _ in range(0,NL):
    iir=[]
    jjr=[]
    for __ in range (0, R): 
        'passo dalle coordinate polari a quelle cartesiane'
        x,y = pol2cart(rho[__],theta[_])
        'centro la origine delle linee nel centro della lesione che ho dato in imput (center_x, center_y)'
        iir.append(center[0]+round(x))
        jjr.append(center[1]+round(y))

    'creo una tabella cioè vettori messi in verticale'
    line1=np.column_stack((iir,jjr))

    #creo matrice di zeri'
    #ROI_lines=np.zeros(np.shape(line1))

    'ho creato una matrice (futura maschera) di zeri'
    Ray_mask=np.zeros(np.shape(ROI))

    for ___ in range(0,len(line1)):
        i=int(line1[___][0])
        j=int(line1[___][1])
        Ray_mask[i,j]=1

    Ray_masks+=Ray_mask

plt.figure()
plt.imshow(Ray_masks)
plt.show()

#%% max variance points
'''funsione che binarizza img'''
def imbinarize(img):
    thresh = threshold_otsu(img)
    img[img >= threshold] = 1
    img[img < threshold] = 0
    return img

J = generic_filter(im_norm, np.std, size=size_nhood_variance)
plt. figure()
plt.imshow(J, cmap='gray')

B_points=[]
roughborder=np.zeros(np.shape(normalized))
'''
Jmasked=J*Ray_masks     #J*raggi=maschera dell'img
Jmasked=Jmasked*imbinarize(ROI)
w = np.where(Jmasked==np.max(Jmasked))    #mi devo accertare che prenda gli indici giusti
'''prova a far girare tuttoil programam per essere sicura che funxioni bene '''
list=[w(0), w(1) , J[w(0),w(1)]]
B_points.extend(list)
roughborder[w(0),w(1)]=normalized[w(0),w(1)]  #copio pixel img all'interno della matrice


#come posso chiudere i bordi ?!
'''
