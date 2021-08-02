import matplotlib.pyplot as plt 
import numpy as np 
%matplotlib inline 
def imshow(img): 
    img = img/2 + 0.5 
    npimg = img.numpy() 
    plt.imshow(np.transpose(npimg, (1,2,0))) 
    plt.show() 
