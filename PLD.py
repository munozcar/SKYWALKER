import numpy as np

def normalizeIntensities (intesityArray):
    '''
    Args:
        intesityArray: array of arrays. There is one array for each time (t). Every
        array includes the intensity of pixels 1 to N at time t.
    Returns:
        normIntensities: array of arrays. The normalized intenisities in intesityArray.
    '''
    normIntensities = []
    for epoch in enumerate(intesityArray):
        currIntensities = intesityArray[epoch[0]]
        numPixels = len(currIntensities)
        normIntensities.append(np.array([i/sum(currIntensities) for i in currIntensities]))
    return normIntensities
