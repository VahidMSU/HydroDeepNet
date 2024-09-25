def NSE(s,o):
    import numpy as np
    s=np.array(s)
    o=np.array(o)
    NSE=1-np.sum((o-s)**2)/np.sum((o-np.mean(o))**2)
    return (NSE)
def R2(s,o):
    import numpy as np
    s=np.array(s)
    o=np.array(o)
    correlation_matrix = np.corrcoef(s, o)
    correlation_xy = correlation_matrix[0,1]
    R2 = correlation_xy**2
    return(R2)
def Pbias(s,o):
    import numpy as np
    s=np.array(s)
    o=np.array(o)
    Pbias=100*sum(s-o)/(sum(o))
    return(Pbias)
def KGE(s,o, ST, Sa, Sb):
    import numpy as np
    s=np.array(s)
    o=np.array(o)
    r=R2(s,o)
    alpha = np.std(s)/np.std(o)
    beta = np.mean(s)/np.mean(o)
    KGE = 1- np.sqrt((ST*(r-1))**2 +(Sa*(alpha-1))**2 +(Sb*(beta-1))**2 )
    return(KGE)
def MAE(s,o):
    import numpy as np
    s=np.array(s)
    o=np.array(o)
    difference_array = abs(np.subtract(s, o))
    MAE = np.mean(difference_array)
    return(MAE)
def MSE(s,o):
    import numpy as np
    difference_array = np.subtract(s, o)
    MSE = np.square(difference_array)
    MSE = MSE.mean()
    return(MSE)
def RMSE(s,o):
    import numpy as np
    difference_array = np.subtract(s, o)
    RMSE = np.square(difference_array)
    RMSE = (RMSE.mean())**0.5
    return(RMSE)
def RSR(s,o):
    import numpy as np
    difference_array = np.subtract(s, o)
    RMSE = np.square(difference_array)
    RMSE = (RMSE.mean())**0.5
    RSR=RMSE/(np.std(o))
    return(RSR)
def SE(s,o):
    import numpy as np
    difference_array = abs(np.subtract(s, o))
    SE = np.square(difference_array)
    SE = (np.sum(difference_array))**(0.5)
    return(SE)
def RE(s,o):
    import numpy as np
    s=np.array(s)
    o=np.array(o)
    RE = sum(abs(np.subtract(s, o)))
    return(RE)