import numpy as np
from PIL import Image


def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15, H_img=False, E_img=False):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(np.float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    _, eigenVector = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigenVector *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    T_hat = ODhat.dot(eigenVector[:,1:3])
    
    phi = np.arctan2(T_hat[:,1],T_hat[:,0])
    
    min_Phi = np.percentile(phi, alpha)
    max_Phi = np.percentile(phi, 100-alpha)
    
    v1 = eigenVector[:,1:3].dot(np.array([(np.cos(min_Phi), np.sin(min_Phi))]).T)
    v2 = eigenVector[:,1:3].dot(np.array([(np.cos(max_Phi), np.sin(max_Phi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if v1[0] > v2[0]:
        HE_vectors = np.array((v1[:,0], v2[:,0])).T
    else:
        HE_vectors = np.array((v2[:,0], v1[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE_vectors,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)
    
    H = None
    if H_img:
        # unmix hematoxylin and eosin
        H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
        H[H>255] = 254
        H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    E = None
    if E_img:
        E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
        E[E>255] = 254
        E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile+'.png')
        if H_img:
            Image.fromarray(H).save(saveFile+'_H.png')
        if E_img:
            Image.fromarray(E).save(saveFile+'_E.png')

    return Inorm, H, E
