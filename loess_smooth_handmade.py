import numpy as np

def loess_smooth_handmade(data, fc, step=1, t=np.nan, t_final=np.nan):
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loess filtering of a time serie
    %
    % data_smooth = loess_smooth_handmade(data, fc)
    %
    % IN:
    %       - data      : time serie to filter
    %       - fc        : cut frequency
    % OPTIONS:
    %       - step      : step between two samples  (if regular)
    %       - t         : coordinates of input data (if not regular)
    %       - t_final   : coordinates of output data (default is t)
    %
    % OUT:
    %       - data_smooth : filtered data
    %
    % Written by R. Escudier (2018)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
  
    if np.all(np.isnan(t)):
        t = np.arange(0, len(data)*step, step)
    
    
    if np.all(np.isnan(t_final)):
        t_final = t
  
    # Remove NaNs
    id_nonan = np.where(~np.isnan(data))
    t = t[id_nonan]
    data = data[id_nonan]
  
    # Period of filter
    tau = 1/fc
    
    # Initialize output vector
    data_smooth = np.ones(t_final.shape)*np.nan
 
    # Only compute for the points where t_final is in the range of t
    sx = np.where(np.logical_and(t_final >= t.min(), t_final <= t.max()))
 
    # Loop on final coordinates
    for i in sx[0]:
        # Compute distance between current point and the rest
        dn_tot = np.abs(t-t_final[i])/tau
        # Select neightboring points
        idx_weights = np.where(dn_tot<1)
        n_pts = len(idx_weights[0])
    
        # Only try to adjust the polynomial if there are at least 4 neighbors
        if n_pts > 3:
            dn = dn_tot[idx_weights]
            w = 1-dn*dn*dn
            weights = w**3
            # adjust a polynomial to these data points
            X = np.stack((np.ones((n_pts,)),t[idx_weights],t[idx_weights]**2)).T
            W = np.diag(weights)
            B = np.linalg.lstsq(np.dot(W,X),np.dot(W,data[idx_weights]))
            coeff = B[0]
            # Smoothed value is the polynomial value at this location
            data_smooth[i] = coeff[0]+coeff[1]*t_final[i]+coeff[2]*t_final[i]**2
        
    return data_smooth