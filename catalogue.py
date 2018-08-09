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

import lime
import lime.lime_tabular
import lime as lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import random
import re

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from keras.metrics import categorical_accuracy

import pylab as P
from scipy.optimize import curve_fit
from matplotlib import gridspec
import pandas as pd

from IPython.display import display, HTML
import pickle

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
    inputs['dg'] = df_carlo['dg'].values #
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
        pl=int((180.0-l[i])/pix_size)
        EBV[i]=EBV_map[pb,pl]
        #print l[i],b[i],pl,pb,EBV_map[pb,pl]
    return EBV

def normalize(df, df_1 = '', kind='std'):
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
           
    # Apply a normalization from training set onto test set. 
    if kind == 'custom':
        for feature_name in df_1.columns:
            if feature_name != 'logg_binary' and feature_name != 'logg_trinary':
                std_value = df_1[feature_name].std()
                mean_value = df_1[feature_name].mean()
                # Apply std and mean from train set onto test set. For globular clusters!
                result[feature_name] = (df[feature_name] - mean_value) / std_value
            
    return result

def change_to_colour(input_test, input_train_col):
    """
    Make data to train on colours and re normalize
    Returns: Normalized colors/input features
    """
    inputs_col = pd.DataFrame()

    inputs_col['u-g']=input_test['u']-input_test['g']
    inputs_col['g-r']=input_test['g']-input_test['r']
    inputs_col['r-i']=input_test['r']-input_test['i']
    inputs_col['i-z']=input_test['i']-input_test['z']
    inputs_col['z-y']=input_test['z']-input_test['y']
    inputs_col['u-G']=input_test['u']-input_test['G']
    inputs_col['u-RP']=input_test['u']-input_test['RP']
    inputs_col['u-BP']=input_test['u']-input_test['BP']
    
    inputs_col = normalize(inputs_col, df_1 = input_train_col, kind='custom')

    return inputs_col
 
def criteria_function(inputs_NGC, std_x, col_1, col_2):
    """
    Inputs_NGC: Inputs of globular cluster
    std_x: The deviation from proper motions to cut
    col_1: lefthand color cut of CMD
    col_2: righthang color cutof CMD
    Apply cuts to the globular cluster catalog in proper motion. Also apply basic color cuts to show CMD better
    Returns the criteria as laid out by parameters
    """
    ra_cut = (inputs_NGC['pmra']>(inputs_NGC['pmra'].mean()-std_x*(inputs_NGC['pmra'].std()))) & (inputs_NGC['pmra']<(inputs_NGC['pmra'].mean()+std_x*(inputs_NGC['pmra'].std())))
    dec_cut = (inputs_NGC['pmdec']>(inputs_NGC['pmdec'].mean()-std_x*(inputs_NGC['pmdec'].std()))) & (inputs_NGC['pmdec']<(inputs_NGC['pmdec'].mean()+std_x*(inputs_NGC['pmdec'].std()))) 
   
    criteria = ra_cut & dec_cut & (inputs_NGC['u']-inputs_NGC['G'] < col_1) & (inputs_NGC['u']-inputs_NGC['G'] > col_2)
    
    return criteria

def prob_frac(bin_width, inputs_NGC):
    """
    Get the luminosity probability function
    """
    bin_loc = []
    avg_prob_dwarf = []
    avg_prob_giant = []
    for data in inputs_NGC['G'].values:
        temp_df = inputs_NGC[(inputs_NGC['G'] < data+bin_width)&(inputs_NGC['G'] > data-bin_width)]
        avg_prob_dwarf.append(temp_df['dwarf'].mean())
        avg_prob_giant.append(temp_df['giant'].mean())
        bin_loc.append(data)
    return avg_prob_dwarf, avg_prob_giant, bin_loc

def lum_frac(bin_width, inputs_NGC):
    """
    Get the luminosity fraction of dwarfs/giants to total
    """
    fraction_giant = []
    fraction_dwarf = []
    bin_loc = []
    for data in inputs_NGC['G'].values:
        temp_df = inputs_NGC
        temp_df_giant = inputs_NGC[(inputs_NGC['G'] < data+bin_width)&(temp_df['G'] > data-bin_width) & (inputs_NGC['class_binary']==1)]
        temp_df_dwarf = inputs_NGC[(inputs_NGC['G'] < data+bin_width)&(temp_df['G'] > data-bin_width) & (inputs_NGC['class_binary']==0)]
        temp_df_all = inputs_NGC[(inputs_NGC['G'] < data+bin_width)&(temp_df['G'] > data-bin_width)]
        fraction_giant.append(float(temp_df_giant.shape[0])/temp_df_all.shape[0])
        fraction_dwarf.append(float(temp_df_dwarf.shape[0])/temp_df_all.shape[0])
        bin_loc.append(data)
    return bin_loc, fraction_giant, fraction_dwarf

def lime_synthesizer(rf, inputf_test, inputf_train, N, binary_class, frac=0.05):
    
    """
    Parameters: inputf_test, inputf_train, N, binary_class, frac=0.05
    inputf_test: testing inputs 
    inputf_train: training inputs
    N: Number of times to get fraction of data and find important features
    binary_class: 0 for dwarf, 1 for giant
    frac: 5% by default. Keep low! 
    
    Returns: A printed dataframe that contains the median important values given by lime
    """
    inputf_test['class_pred'] = rf.predict(inputf_test)
    
    explainer = lime.lime_tabular.LimeTabularExplainer(inputf_train.values, feature_names = inputf_train.columns, discretize_continuous=True)
    
    for i in range(0, N):
        df = inputf_test[inputf_test['class_pred']==binary_class]
        df = df.drop(columns=['class_pred'])
        df = df.sample(frac=frac)
        ints = range(0, df.shape[0])
        print i

        lime_vals = []
        for index in ints:
            print index, 'of', len(ints)
            exp = explainer.explain_instance(df.values[index], rf.predict_proba, num_features=inputf_train.columns.shape[0])
            lime_vals.append(exp.as_list())

        array = np.asarray(lime_vals)

        # This is only for dwarfs keep in mind
        low_a=[]
        high_a=[]
        feature_a=[]
        val_a = []
        for i in range(0, array.shape[0]):
            for j in range(0, array.shape[1]):
                instance = re.split("< | <= | >", array[i, j][0])
                if len(re.split("< | <= | >", array[i, j][0]))==2:
                    feature = instance[0]
                    low = instance[1]
                    val = array[i,j,1]

                    val_a.append(val)
                    feature_a.append(feature)
                    low_a.append(low)
                    high_a.append('NaN')
                if len(re.split("< | <= | >", array[i, j][0]))==3:
                    low = instance[0]
                    feature = instance[1]
                    high = instance[2]
                    val = array[i,j,1]

                    val_a.append(val)
                    feature_a.append(feature)
                    low_a.append(low)  
                    high_a.append(high)

        df_lime_trans = pd.DataFrame()
        df_lime_trans['low'] = low_a
        df_lime_trans['high'] = high_a
        df_lime_trans['feature'] = feature_a
        df_lime_trans['val'] = val_a

        grouped = df_lime_trans.groupby('feature')

        df_lime = pd.DataFrame()

        p={}
        panel= pd.Panel(p)
        for name,group in grouped:
            p[name] = group
        panel = pd.Panel(p)

        low_median = [panel[name]['low'].dropna().astype(float).median() for name, group in grouped]
        high_median = [panel[name]['high'].dropna().astype(float).median() for name, group in grouped]
        val_median = [panel[name]['val'].dropna().astype(float).median() for name, group in grouped]
        name = [name for name,group in grouped ]

        df_lime['low'] = low_median
        df_lime['feature'] = name
        df_lime['high'] = high_median
        df_lime['val'] = val_median

        print df_lime.sort_values(by=['val'])
