from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_and_center_dataset(filename):
     
    x = np.load(filename)
    
    mean_x = np.mean(x, axis=0)
    
    return x - mean_x

def get_covariance(dataset):
    
    x = dataset
    
    xT = np.transpose(x)
    
    covariance_x = np.dot(xT, x) * (1 / (len(x) - 1))

    return covariance_x
    
def get_eig(S, m):
    
    l = len(S)
    w, v = eigh(S, subset_by_index=[l - m, l - 1])
    
    w = np.diag(np.flip(w))
    
    v[:, [0,1]] = v[:, [1,0]]
    
    return w, v

def get_eig_prop(S, prop):
    
    w, v = eigh(S)
    
    w_matrix = np.diag(np.flip(w))
    
    w_sum = np.trace(w_matrix)
    
    count = 0
    for eig_val in np.flip(w):
        if (eig_val / w_sum) >= prop:
            count += 1
            continue
        else:
            break
        
    return get_eig(S, count)    

def project_image(image, U):
    
    xi = 0
    
    for col in np.transpose(U):
        uT = np.transpose(col)
        aij = np.dot(uT, image)
        xi += np.dot(aij, col)
    
    return xi

def display_image(orig, proj):
    
    orig = np.reshape(orig, (32,32))
    proj = np.reshape(proj, (32,32))
    
    fig, a = plt.subplots(1, 2)
    color1 = a[0].imshow(np.transpose(orig), aspect='equal')
    color2 = a[1].imshow(np.transpose(proj), aspect='equal')
    a[0].set_title("Original")
    a[1].set_title("Projection")
    
    divider1 = make_axes_locatable(a[0])
    divider2 = make_axes_locatable(a[1])
    
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
      
    fig.colorbar(color1, ax=a[0], cax=cax1)
    fig.colorbar(color2, ax=a[1], cax=cax2)
    
    fig.tight_layout(pad=2)
    
    plt.show()
    