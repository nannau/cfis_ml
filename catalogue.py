import numpy as np
from astropy.io import fits
from astropy import units
from astropy.coordinates import SkyCoord
from astropy import units
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn import preprocessing

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Activation, multiply
from keras.engine.topology import Layer
from keras.optimizers import Adam
from keras import backend as K

import pylab as P
from scipy.optimize import curve_fit
from matplotlib import gridspec
import pandas as pd

from IPython.display import display, HTML
import pickle

class Predictor_weighted_loss_layer(Layer):
    __name__ = u'pred_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(Predictor_weighted_loss_layer, self).__init__(**kwargs)

    def lossfun(self, y_true, y_pred, y_err):
        #mae_loss = K.mean(K.abs((y_true - y_pred))/(y_err+0.01))
        mae_loss = K.mean(K.abs((y_true - y_pred)))
        return mae_loss

    def call(self, inputs):
        y_true = inputs[0]
        y_pred = inputs[1]
        y_err = inputs[2]
        loss = self.lossfun(y_true, y_pred, y_err)
        self.add_loss(loss, inputs=inputs)

        return y_true
    
# this is used to compile the model, returning a zero-loss so no gradients are returned by the regular keras way of
# analyzing a loss function. The weighted loss above will be the only one that matters
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)


class catalogue(object):
    """
    Condensing code to draw from fits catalogue of Milky Way stars for machine learning.
    
    Will be using panstarrs, gaia, cfis, and sequey for this extraction.
    """

    def __init__(self, ra, dec, u, g, r, i, z, y, du, dg, dr, di, dz, dy, Teff, logg, feh, dTeff, dlogg, dfeh):
        """Return a catalogue's physical parameters."""
        self.ra = ra
        self.dec = dec
        self.u = u
        self.g = g
        self.r = r
        self.i = i
        self.z = z
        self.y = y
        self.du = du
        self.dg = dg
        self.dr = dr
        self.di = di
        self.dz = dz
        self.dy = dy
        self.Teff = Teff
        self.logg = logg
        self.feh = feh
        self.dTeff = dTeff
        self.dlogg = dlogg
        self.dfeh = dfeh
        

def load_data(filename='./data/cfis_ps_segue_gaia.fits', dust_filename='./data/polecount_dust.fits'):

    """
    Function to load data from stellar catalogue
    - Add values to pandas dataframe
    - Applies a filter to get the best data
    - Applies dust correction 
    - Runs a monte carlo simulation on data to create 10*(Number of stars)
    - Separates parameters into inputs, outputs, and errors for machine learning.
    =============================================
    
    Input: Stellar catalogue, dust file (fits)
    
    Output: 4 monte-carlo'd dataframes with inputs, outputs, error, and all parameters
    """

    hdulist = fits.open(filename)
    data=hdulist[1].data        
    ra=data.field(0)
    dec=data.field(1)
    u=data.field(2) #u
    g=data.field(10) #g
    r=data.field(12) #r
    i=data.field(14) #i
    z=data.field(16) #z
    y=data.field(18) #y
    G=data.field(91) #G
    BP=data.field(93) #BP
    RP=data.field(95) #RP
    du=data.field(3) #du
    dg=data.field(11) #dg
    dr=data.field(13) #dr
    di=data.field(15) #di
    dz=data.field(17) #dz
    dy=data.field(19) #dy
    nG=data.field(90) #nobs G
    nBP=data.field(92) #nobs BP
    nRP=data.field(94) #nobs RP


    Teff=data.field(58) #Teff
    logg=data.field(60) #log(g)
    feh=data.field(62)  #[Fe/H]
    dTeff=data.field(59) #Teff
    dlogg=data.field(61) #log(g)
    dfeh=data.field(63)  #[Fe/H]
    SNR=data.field(66) #SNR
    hdulist.close()
    
    column_headers = []
    indexes = np.array([ra, dec, u, g, r, i, z, y, G, BP, RP, du, dg, dr, di, dz, dy, Teff, logg, feh, dTeff, dlogg, dfeh])
    
    df= pd.DataFrame()
    

    # loading into a pandas dataframe, but swapping bytes due to strange error.
    df['ra'], df['dec'] = ra.byteswap().newbyteorder(), dec.byteswap().newbyteorder()
    df['g'], df['u'], df['r'] =  g.byteswap().newbyteorder(), u.byteswap().newbyteorder(), r.byteswap().newbyteorder()
    df['i'], df['z'], df['y'] = i.byteswap().newbyteorder(), z.byteswap().newbyteorder(), y.byteswap().newbyteorder()
    df['G'], df['BP'], df['RP'] = G.byteswap().newbyteorder(), BP.byteswap().newbyteorder(), RP.byteswap().newbyteorder()
    df['du'], df['dg'] = du.byteswap().newbyteorder(), dg.byteswap().newbyteorder()
    df['dr'], df['di'], df['dz'] = dr.byteswap().newbyteorder(), di.byteswap().newbyteorder(), dz.byteswap().newbyteorder()
    df['dy'], df['Teff'] = dy.byteswap().newbyteorder(), Teff.byteswap().newbyteorder()
    df['logg'], df['feh'], df['dTeff'] = logg.byteswap().newbyteorder(), feh.byteswap().newbyteorder(), dTeff.byteswap().newbyteorder()
    df['dlogg'], df['dfeh'] = dlogg.byteswap().newbyteorder(), dfeh.byteswap().newbyteorder()
    df['SNR'], df['nG'], df['nBP'], df['nRP'] = SNR.byteswap().newbyteorder(), nG.byteswap().newbyteorder(), nBP.byteswap().newbyteorder(), nRP.byteswap().newbyteorder()

    # Cut in uncertainties and select only stars having gaia photometry and apply it to each column
    dTeff_thres = 10000.0 #120.0
    dlogg_thres = 10000.0 #0.13
    dfeh_thres = 10000.0 #0.2
    SNR_thres = 50.0 
    FeH_thres = -40.0
        
    criteria = (df['dTeff'] <=dTeff_thres) & (df['dlogg']<=dlogg_thres) & (df['dfeh']<=dfeh_thres) & (df['SNR']>=SNR_thres) & (df['feh']>=FeH_thres) &(df['nG']>0) &(df['nBP']>0) &(df['nRP']>0)
    
    df_filtered = df[criteria]
    df_filtered.reset_index()
    
    # Do a monte-carlo sample
    df_carlo = monte_carlo(df_filtered)
    df_carlo.reset_index(drop=True)
    # Calculate the extinction
    EBV=get_EBV(dust_filename, df_carlo['ra'],df_carlo['dec'])
    
    # Generate the inputs and outputs catalogues
    inputs = pd.DataFrame()
    inputs_col = pd.DataFrame()
    outputs = pd.DataFrame()
    error = pd.DataFrame()

    # Dered data for input
    inputs['u'] = df_carlo['u'].values - 4.239*EBV # I'm pretending these are SDSS band for u, but it is not and PS for the others!
    inputs['g'] = df_carlo['g'].values - 3.172*EBV #
    inputs['r'] = df_carlo['r'].values - 2.271*EBV #
    inputs['i'] = df_carlo['i'].values - 1.682*EBV #
    inputs['z'] = df_carlo['z'].values - 1.322*EBV #
    inputs['y'] = df_carlo['y'].values - 1.087*EBV #
    inputs['G'] = df_carlo['G'].values - 0.85926*EBV # Assume extinction coefficiants from Malhan+ 2018b
    inputs['BP'] = df_carlo['BP'].values - 1.06794*EBV
    inputs['RP'] = df_carlo['RP'].values - 0.65199*EBV
    
    outputs['Teff'] = df_carlo['Teff']
    outputs['logg'] = df_carlo['logg']
    outputs['feh'] = df_carlo['feh']

    error['dTeff'] = df_carlo['dTeff']
    error['dlogg'] = df_carlo['dlogg']
    error['dfeh'] = df_carlo['dfeh']

    return inputs, outputs, error, df_carlo

def load_data_glob(filename='./data/cfis_ps_segue_gaia.fits', dust_filename='./data/polecount_dust.fits'):

    """
    Function to load data from stellar catalogue
    - Add values to pandas dataframe
    - Applies a filter to get the best data
    - Applies dust correction 
    - Runs a monte carlo simulation on data to create 10*(Number of stars)
    - Separates parameters into inputs, outputs, and errors for machine learning.
    =============================================
    
    Input: Stellar catalogue, dust file (fits)
    
    Output: 4 monte-carlo'd dataframes with inputs, outputs, error, and all parameters
    """

    hdulist = fits.open(filename)
    data=hdulist[1].data        
    ra=data.field(0)
    dec=data.field(1)
    u=data.field(2) #u
    g=data.field(10) #g
    r=data.field(12) #r
    i=data.field(14) #i
    z=data.field(16) #z
    y=data.field(18) #y
    G=data.field(66) #G
    BP=data.field(68) #BP
    RP=data.field(70) #RP
    du=data.field(3) #du
    dg=data.field(11) #dg
    dr=data.field(13) #dr
    di=data.field(15) #di
    dz=data.field(17) #dz
    dy=data.field(19) #dy
    nG=data.field(67) #nobs G
    nBP=data.field(69) #nobs BP
    nRP=data.field(71) #nobs RP
    pmdec = data.field(53) # proper motion
    pmra = data.field(51) # proper motion

    hdulist.close()
    
    column_headers = []
 
    df= pd.DataFrame()

    # loading into a pandas dataframe, but swapping bytes due to strange error.
    df['ra'], df['dec'] = ra.byteswap().newbyteorder(), dec.byteswap().newbyteorder()
    df['g'], df['u'], df['r'] =  g.byteswap().newbyteorder(), u.byteswap().newbyteorder(), r.byteswap().newbyteorder()
    df['i'], df['z'], df['y'] = i.byteswap().newbyteorder(), z.byteswap().newbyteorder(), y.byteswap().newbyteorder()
    df['G'], df['BP'], df['RP'] = G.byteswap().newbyteorder(), BP.byteswap().newbyteorder(), RP.byteswap().newbyteorder()
    df['du'], df['dg'] = du.byteswap().newbyteorder(), dg.byteswap().newbyteorder()
    df['dr'], df['di'], df['dz'] = dr.byteswap().newbyteorder(), di.byteswap().newbyteorder(), dz.byteswap().newbyteorder()
    df['dy'] = dy.byteswap().newbyteorder()
    df['nG'], df['nBP'], df['nRP'] = nG.byteswap().newbyteorder(), nBP.byteswap().newbyteorder(), nRP.byteswap().newbyteorder()
    df['pmra'], df['pmdec'] = pmra.byteswap().newbyteorder(), pmdec.byteswap().newbyteorder()
         
    #criteria = (df['nG']>0) &(df['nBP']>0) &(df['nRP']>0)
    
    df_filtered = df#[criteria]
    df_filtered.reset_index()
    
    # Do a monte-carlo sample
    df_carlo = df_filtered#monte_carlo_glob(df_filtered)
    df_carlo.reset_index(drop=True)
    # Calculate the extinction
    EBV=get_EBV(dust_filename, df_carlo['ra'],df_carlo['dec'])
    
    # Generate the inputs and outputs catalogues
    inputs = pd.DataFrame()

    # Dered data for input
    inputs['u'] = df_carlo['u'].values - 4.239*EBV # I'm pretending these are SDSS band for u, but it is not and PS for the others!
    inputs['g'] = df_carlo['g'].values - 3.172*EBV #
    inputs['r'] = df_carlo['r'].values - 2.271*EBV #
    inputs['i'] = df_carlo['i'].values - 1.682*EBV #
    inputs['z'] = df_carlo['z'].values - 1.322*EBV #
    inputs['y'] = df_carlo['y'].values - 1.087*EBV #
    inputs['G'] = df_carlo['G'].values - 0.85926*EBV # Assume extinction coefficiants from Malhan+ 2018b
    inputs['BP'] = df_carlo['BP'].values - 1.06794*EBV
    inputs['RP'] = df_carlo['RP'].values - 0.65199*EBV
    
    inputs['ra'] = df_carlo['ra']
    inputs['dec'] = df_carlo['dec']
    inputs['pmra'], inputs['pmdec'] = df_carlo['pmra'], df_carlo['pmdec']
    
    return inputs

def monte_carlo_glob(df):
    """
    Takes a pandas dataframe and runs a monte-carlo simulation on the target 
    labels.
    ====================================================================
    Inputs: Pandas df
    
    Outputs: Monte-carlo'd pandas df
    """
    
    nb_increase=10 # Number of montecarlo sample 
    size = len(df['u'])
    df_old = df

    for nb in range(1,nb_increase):
        df_tmp = df_old
        df_n = pd.DataFrame()
        df_n['u'] = df_old['u'].values + np.random.normal(0.0, 1.0, size)*df_old['du'].values
        df_n['g'] = df_old['g'].values + np.random.normal(0.0, 1.0, size)*df_old['dg'].values
        df_n['r'] = df_old['r'].values + np.random.normal(0.0, 1.0, size)*df_old['dr'].values
        df_n['i'] = df_old['i'].values + np.random.normal(0.0, 1.0, size)*df_old['di'].values
        df_n['z'] = df_old['z'].values + np.random.normal(0.0, 1.0, size)*df_old['dz'].values
        df_n['y'] = df_old['y'].values + np.random.normal(0.0, 1.0, size)*df_old['dy'].values

        df_tmp.update(df_n)
        
        df = pd.concat([df, df_tmp], ignore_index=True)

    return df

def monte_carlo(df):
    """
    Takes a pandas dataframe and runs a monte-carlo simulation on the target 
    labels.
    ====================================================================
    Inputs: Pandas df
    
    Outputs: Monte-carlo'd pandas df
    """
    
    nb_increase=10 # Number of montecarlo sample 
    size = len(df['u'])
    df_old = df

    for nb in range(1,nb_increase):
        df_tmp = df_old
        df_n = pd.DataFrame()
        df_n['u'] = df_old['u'].values + np.random.normal(0.0, 1.0, size)*df_old['du'].values
        df_n['g'] = df_old['g'].values + np.random.normal(0.0, 1.0, size)*df_old['dg'].values
        df_n['r'] = df_old['r'].values + np.random.normal(0.0, 1.0, size)*df_old['dr'].values
        df_n['i'] = df_old['i'].values + np.random.normal(0.0, 1.0, size)*df_old['di'].values
        df_n['z'] = df_old['z'].values + np.random.normal(0.0, 1.0, size)*df_old['dz'].values
        df_n['y'] = df_old['y'].values + np.random.normal(0.0, 1.0, size)*df_old['dy'].values
        
        df_n['Teff'] = df_old['Teff'].values + np.random.normal(0.0, 1.0, size)*df_old['dTeff'].values        
        df_n['logg'] = df_old['logg'].values + np.random.normal(0.0, 1.0, size)*df_old['dlogg'].values
        df_n['feh'] = df_old['feh'].values + np.random.normal(0.0, 1.0, size)*df_old['dfeh'].values

        df_tmp.update(df_n)
        
        df = pd.concat([df, df_tmp], ignore_index=True)

    return df


def get_EBV(dust_filename, ra, dec):
    """
    Function for getting the extinction coefficient of the star.
    ===========================================================
    inputs: Dust fits file, ra, dec dataframe column
    outputs: Dust coefficient for each star in dataframe
    """
    ra = ra.values
    dec = dec.values
    # Transformation of coordinate
    c_icrs = SkyCoord(ra=ra*units.degree, dec=dec*units.degree, frame='icrs')
    l=c_icrs.galactic.l.degree
    b=c_icrs.galactic.b.degree
    # Read the map of schlegel et al., 98
    EBV=np.zeros(len(ra))
    ebvlist = fits.open(dust_filename)
    EBV_map= ebvlist[0].data
    ebvlist.close()
    pix_size=0.1 # Degree
    for i in range(0,len(ra)):
        pb=int((b[i]+90.0)/pix_size) 
        pl=int(l[i]/pix_size)
        EBV[i]=EBV_map[pb,pl]
        #print l[i],b[i],pl,pb,EBV_map[pb,pl]
    return EBV

def ML2_get_logg(inputs,outputs):
    """
    Logg prediction NN
    """
    
    input_shape = inputs.shape[1]
    output_shape = 1
    initializer='he_normal'
    
    inputs = Input(shape=(input_shape,))
    
    x = Dense(32,kernel_initializer=initializer)(inputs)
    x = Activation(u'relu')(x)
    x = BatchNormalization()(x)
    
    x = Dense(5096,kernel_initializer=initializer)(x)
    x = Activation(u'relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(256,kernel_initializer=initializer)(x)
    x = Activation(u'relu')(x)
    x = BatchNormalization()(x)    
    
    x = Dense(256,kernel_initializer=initializer)(x)
    x = Activation(u'relu')(x)
    x = BatchNormalization()(x)    

def normalize(df, kind='std'):
    """
    Function to normalize pandas dataframe values. 
    Can be max_min or the standard deviation, mean method 
    ==============================================
    inputs: df to normalize, kind of normalization
    outputs: normalized df
    """
    
    result = df.copy()
    if kind == 'std':
        for feature_name in df.columns:
            if feature_name != 'logg_binary' and feature_name != 'logg_trinary':
                std_value = df[feature_name].std()
                mean_value = df[feature_name].mean()
                result[feature_name] = (df[feature_name] - mean_value) / std_value

    if kind == 'min_max':
        for feature_name in df.columns:
            if feature_name != 'logg_binary' and feature_name != 'logg_trinary':
                max_value = df[feature_name].max()
                min_value = df[feature_name].min()
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            
    return result
