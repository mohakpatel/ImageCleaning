import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

def seedBeadsN(pts, sizeI, sigma):

    nPts, nDims = pts.shape
    bead_size = np.ceil(sigma*2*5)

    # Create grid points to compute gaussian particle intensity
    if nDims == 2:
        m = np.mgrid[0:bead_size[0], 0:bead_size[1]]
    elif nDims == 3:
        m = np.mgrid[0:bead_size[0], 0:bead_size[1], 0:bead_size[2]]
    else:
        print('Can only handle 2D or 3D image')
        return -1
    
    # Create gaussian intensity profile for each particle in local subset
    relative_pts = (pts - np.floor(pts)) + bead_size/2
    f = [0]*nPts
    for i in range(nPts):
        for j in range(nDims):
            f[i] = f[i] - ((m[j] - relative_pts[i,j]) / (2*sigma[j]))**2
        f[i] = np.exp(f[i])

    # Compute index location in the original image where the local guassian 
    # subset intensity will be joined/added in the original image
    idx , mask = [[None for _ in range(nPts)] for _ in range(nDims)], [0]*nDims
    for i in range(nDims):
        temp = (np.expand_dims(np.floor(pts[:,i]), axis=1) + 
                    np.matlib.repmat(np.arange(bead_size[i]), nPts, 1) -
                    bead_size[i]/2)
        mask[i] = np.logical_and(temp >=0, temp<sizeI[i])
        for j in range(temp.shape[0]):
            idx[i][j] = temp[j,:].astype(int)
        
        # Find indexes which go outside image frame and crop them
        j_ = np.where(np.any(~mask[i], axis=1))[0].tolist()
        if j_:
            for j in j_:
                if nDims == 2:
                    if i == 0:
                        f[j] = f[j][mask[i][j,:],:]
                    elif i == 1:
                        f[j] = f[j][:,mask[i][j,:]]

                elif nDims == 3:
                    if i == 0:
                        f[j] = f[j][mask[i][j,:],:,:]
                    elif i == 1:
                        f[j] = f[j][:,mask[i][j,:],:]
                    elif i == 2:
                        f[j] = f[j][:,:,mask[i][j,:]]
                
                idx[i][j] = idx[i][j][mask[i][j,:]]
    
    # Add image subset for each particle to the main image
    I = np.zeros(sizeI)
    if nDims == 2:
        for i in range(nPts):
        
            I[idx[0][i][0]:idx[0][i][-1]+1, 
              idx[1][i][0]:idx[1][i][-1]+1] = (I[idx[0][i][0]:idx[0][i][-1]+1, 
                                                idx[1][i][0]:idx[1][i][-1]+1] + 
                                                f[i])
    elif nDims == 3:        
        for i in range(nPts):
            
            I[idx[0][i][0]:idx[0][i][-1]+1, 
              idx[1][i][0]:idx[1][i][-1]+1,
              idx[2][i][0]:idx[2][i][-1]+1] = (I[idx[0][i][0]:idx[0][i][-1]+1, 
                                                idx[1][i][0]:idx[1][i][-1]+1,
                                                idx[2][i][0]:idx[2][i][-1]+1] + 
                                                f[i])        

    I[I > 1] = 1
    return I

def random_seed_locations(nPts, sizeI):
    return np.random.random((nPts, len(sizeI)))*sizeI


sigma = np.array([1, 1])
sizeI = np.array([64, 64])
pts = random_seed_locations(100, sizeI)

I = seedBeadsN(pts, sizeI, sigma)
plt.imshow(I)
plt.show()
