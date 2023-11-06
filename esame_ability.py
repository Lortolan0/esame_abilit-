# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:13:37 2023

@author: Checco
"""

from astropy.io import fits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
path = os.getcwd()

TEST_COVARIANCE=False
PLOTS=True
    
Nbins=200
Nmeasures=100 # 10000

nm = ['1', '3', '5']
cn = ['1-3', '1-5', '3-5']
tn = ['1', '2', '3']



# SI RIPETE IL PROGRAMMA PER OGNI SET DI MISURE
for sset in range(3):
    
    
    
    
    # INIZIO DEL PROGRAMMA PER UN SET DI MISURE
    test=sset+1 #SELEZIONE DEL SET DI MISURE

    measures0=[]
    measures2=[]
    measures4=[]
    
    cov_M = []
    cov_T = []
    cross_T = []
    cross_M = []
    
    
    matrix_M = []
    matrix_T = []
    
    
    #big_M = np.zeros((3,3), dtype = float)
    
    # ESTRAZIONE DEI MULTIPOLI
    for i in np.arange(Nmeasures)+1:
        fname = path + f'/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
    
        file = fits.open(fname)
        table = file[1].data.copy()
        measures0.append(table['XI0'])
        measures2.append(table['XI2'])
        measures4.append(table['XI4'])
        
        if i==1:
            scale = table['SCALE']
        del table
        file.close()
    
    
    # TRASPOSIZIONE DELLE MATRICI DI DATI
    measures0=np.asarray(measures0).transpose()
    measures2=np.asarray(measures2).transpose()
    measures4=np.asarray(measures4).transpose()
    
    # CALCOLO DELLE MEDIE E COVARIANZE NUMERICHE
    mean_xi_0 = np.mean(measures0,axis=1)
    mean_xi_2 = np.mean(measures2,axis=1)
    mean_xi_4 = np.mean(measures4,axis=1)
    cov_xi_0 = np.cov(measures0)
    cov_xi_2 = np.cov(measures2)
    cov_xi_4 = np.cov(measures4)
    
    
    
###############################################################################
########## TEST DI COVARIANZA NUMERICA E NUMPY (OPZIONALE)  ###################
###############################################################################
    if TEST_COVARIANCE:
        print('Running test to see if I understand the covariance:')
    
        AVE0 = np.zeros((Nbins,),dtype=float)
        AVE2 = np.zeros((Nbins,),dtype=float)
        AVE4 = np.zeros((Nbins,),dtype=float)
        COV0 = np.zeros((Nbins,Nbins),dtype=float)
        COV2 = np.zeros((Nbins,Nbins),dtype=float)
        COV4 = np.zeros((Nbins,Nbins),dtype=float)
    
        for k in range(Nmeasures):
            AVE0 += measures0[:,k]
            AVE2 += measures0[:,k]
            AVE4 += measures0[:,k]
        AVE0 /= Nmeasures
        AVE2 /= Nmeasures
        AVE4 /= Nmeasures
    
        for i in range(Nbins):
            for j in range(Nbins):
                COV0[i,j] = (np.sum(measures0[i]*measures0[j]) - AVE0[i]*AVE0[j]*Nmeasures) / (Nmeasures-1)
                COV2[i,j] = (np.sum(measures2[i]*measures2[j]) - AVE2[i]*AVE2[j]*Nmeasures) / (Nmeasures-1)
                COV4[i,j] = (np.sum(measures4[i]*measures4[j]) - AVE4[i]*AVE4[j]*Nmeasures) / (Nmeasures-1)
    
        print('Largest deviation between my calculation and numpy in multipole 1: {}'.format(np.max(np.abs(COV0-cov_xi_0))))
        print('Largest deviation between my calculation and numpy in multipole 3: {}'.format(np.max(np.abs(COV2-cov_xi_2))))
        print('Largest deviation between my calculation and numpy in multipole 5: {}'.format(np.max(np.abs(COV4-cov_xi_4))))

###############################################################################
###############################################################################    
    
    # RAGGRUPPO LE COVARIANZE MISURATE IN UN SINGOLO ARRAY
    cov_M.append(cov_xi_0)
    cov_M.append(cov_xi_2)
    cov_M.append(cov_xi_4)
    
    # MATRICI DI CORRELAZIONE
    corr_xi_0 = np.zeros((Nbins,Nbins),dtype=float)
    corr_xi_2 = np.zeros((Nbins,Nbins),dtype=float)
    corr_xi_4 = np.zeros((Nbins,Nbins),dtype=float)
    for i in range(Nbins):
        for j in range(Nbins):
            corr_xi_0[i,j]=cov_xi_0[i,j]/(cov_xi_0[i,i]*cov_xi_0[j,j])**0.5
            corr_xi_2[i,j]=cov_xi_2[i,j]/(cov_xi_2[i,i]*cov_xi_2[j,j])**0.5
            corr_xi_4[i,j]=cov_xi_4[i,j]/(cov_xi_4[i,i]*cov_xi_4[j,j])**0.5
    
    
    # PARAMETRI DEL SET DI MISURE
    if test==1:
        sigs = [0.02, 0.02, 0.02]
        ls = [25, 50, 75]
    elif test==2:
        sigs = [0.02, 0.01, 0.005]
        ls = [50, 50, 50]
    else:
        sigs = [0.02, 0.01, 0.005]
        ls = [5, 5, 5]
    
    
###############################################################################
#######   DEFINIZIONE DELLE MATRICI DI CORRELAZIONE E CORRELAZIONE MISTA ######
###############################################################################   
    
    ## Definitions to build the covarince matrices based on Squared Exponential kernel
    def covf(x1, x2, sig, l):
        return sig**2.*np.exp(-(x1 - x2)**2./(2.*l**2.))
    
    def covf1f2(x1, x2, sig1, l1, sig2, l2):
        return (np.sqrt(2.*l1*l2)*np.exp(-(np.sqrt((x1 - x2)**2.)**2./(l1**2. + l2**2.)))*sig1*sig2)/np.sqrt(l1**2. + l2**2.)
    
    
    #   DEFINIZIONE DELLE MATRICI DI CORRELAZIONE MISTA NUMERICHE
    def cov_num_cr(mat_0,mat_2):
        rows,cols = mat_0.shape
        average_0 = np.zeros((rows,),dtype=float)
        average_2 = np.zeros((rows,),dtype=float)
        covariance = np.zeros((rows,rows),dtype=float)
        
        for i in range(cols):
            average_0 += mat_0[:,i]
            average_2 += mat_2[:,i]
        average_0 /= cols                     # average = average / cols
        average_2 /= cols                     # average = average / cols
    
        
        for i in range(rows):
            for j in range(rows):
                covariance[i,j] = (np.sum(mat_0[i]*mat_2[j]) - average_0[i]*average_2[j]*cols) / (cols-1)
                
                #COV4[i,j] = (np.sum(measures4[i]*measures4[j]) - AVE4[i]*AVE4[j]*Nmeasures) / (Nmeasures-1)
        
        return covariance
    
###############################################################################
###############################################################################


    # CALCOLO DELLE COVARIANZE TEORICHE
    cov_th_0 = np.zeros((Nbins,Nbins),dtype=float)
    cov_th_2 = np.zeros((Nbins,Nbins),dtype=float)
    cov_th_4 = np.zeros((Nbins,Nbins),dtype=float)
    for i in range(Nbins):
        for j in range(Nbins):
            cov_th_0[i,j] = covf(scale[i],scale[j],sigs[0],ls[0])
            cov_th_2[i,j] = covf(scale[i],scale[j],sigs[1],ls[1])
            cov_th_4[i,j] = covf(scale[i],scale[j],sigs[2],ls[2])
    
    
    # RAGGRUPPO LE COVARIANZE TEORICHE IN UN ARRAY
    cov_T.append(cov_th_0)
    cov_T.append(cov_th_2)
    cov_T.append(cov_th_4)
    
    
    
    # CALCOLO DELLE CORRELAZIONI MISTE TEORICHE
    cross_th_02 = np.zeros((Nbins, Nbins), dtype = float)
    cross_th_04 = np.zeros((Nbins, Nbins), dtype = float)
    cross_th_24 = np.zeros((Nbins, Nbins), dtype = float)
    for i in range(Nbins):
        for j in range(Nbins):
            cross_th_02[i][j] = covf1f2(scale[i], scale[j], sigs[0], ls[0], sigs[1], ls[1])
            cross_th_04[i][j] = covf1f2(scale[i], scale[j], sigs[0], ls[0], sigs[2], ls[2])
            cross_th_24[i][j] = covf1f2(scale[i], scale[j], sigs[1], ls[1], sigs[2], ls[2])
    
    
    # RAGGRUPPO LE MATRICI DI CORRELAZIONE MISTA TEORICA IN UN ARRAY
    cross_T.append(cross_th_02)
    cross_T.append(cross_th_04)
    cross_T.append(cross_th_24)

    # CALCOLO LE MATRICI DI CORRELAZIONE MISTA NUMERICHE IN UN ARRAY
    cross_xi_02 = cov_num_cr(measures0, measures2)
    cross_xi_04 = cov_num_cr(measures0, measures4)
    cross_xi_24 = cov_num_cr(measures2, measures4)
    
    # RAGGRUPPO LE MATRICI DI CORRELAZIONE MISTA NUMERICHE IN UN ARRAY
    cross_M.append(cross_xi_02)
    cross_M.append(cross_xi_04)
    cross_M.append(cross_xi_24)
    
    
    
###############################################################################
#################################   PLOTS   ###################################
###############################################################################
    # GRAFICI DELLE COVARIANZE TEORICHE, MISURATE E RESIDUI
    if PLOTS:
    
        gratio = (1. + 5. ** 0.5) / 2.
    
        dpi = 300
        #climit=max(np.max(theoretical_covariance),np.max(measured_covariance))
        cmin = []
        cmax = []
        ccmin = []
        ccmax = []
            
        for i in range(3):
            cmin.append(-np.max(cov_T[i]) * 0.05)
            ccmin.append(-np.max(cross_T[i]) * 0.05)
            cmax.append(np.max(cov_T[i]) * 1.05)
            ccmax.append(np.max(cross_T[i]) * 1.05)
    
        
        # PLOT PER CIASCUN MULTIPOLO E MULTIPOLI MISTI
        for i in range(3):
            # Matrix plot of measured covariance matrix
            fig = plt.figure(figsize=(6,4))
            plt.title('measured covariance matrix '+nm[i]+', set '+tn[sset])
            plt.imshow(cov_M[i], vmin=cmin[i], vmax=cmax[i])
            cbar = plt.colorbar(orientation="vertical", pad=0.02)
            cbar.set_label(r'$ C^{\xi}_{N}$')
            plt.show()
            
        
            # Matrix plot of theoretical covariance matrix
            fig = plt.figure(figsize=(6,4))
            plt.title('theoretical covariance matrix '+nm[i]+', set '+tn[sset])
            plt.imshow(cov_T[i], vmin=cmin[i], vmax=cmax[i])
            cbar = plt.colorbar(orientation="vertical", pad=0.02)
            cbar.set_label(r'$ C^{\xi}_{N}$')
            plt.show()
            
            
            # Matrix plot of residual matrix
            fig = plt.figure(figsize=(6,4))
            plt.title('residuals '+nm[i]+', set '+tn[sset])
            plt.imshow(cov_T[i] - cov_M[i], vmin=cmin[i], vmax=-cmin[i])
            cbar = plt.colorbar(orientation="vertical", pad=0.02)
            cbar.set_label(r'$ C^{\xi}_{N}$')
            plt.show()
            
            
            # Matrix plot of measured mixed covariance matrix
            fig = plt.figure(figsize=(6,4))
            plt.title('measured mixed covariance matrix '+cn[i]+', set '+tn[sset])
            plt.imshow(cross_M[i], vmin=ccmin[i], vmax=ccmax[i])
            cbar = plt.colorbar(orientation="vertical", pad=0.02)
            cbar.set_label(r'$ C^{\xi}_{N}$')
            plt.show()
            
            
            # Matrix plot of theoretical mixed covariance matrix
            fig = plt.figure(figsize=(6,4))
            plt.title('theoretical mixed covariance matrix '+cn[i]+', set '+tn[sset])
            plt.imshow(cross_T[i], vmin=ccmin[i], vmax=ccmax[i])
            cbar = plt.colorbar(orientation="vertical", pad=0.02)
            cbar.set_label(r'$ C^{\xi}_{N}$')
            plt.show()
            
            
            # Matrix plot of residual matrix
            fig = plt.figure(figsize=(6,4))
            plt.title('residuals '+cn[i]+', set '+tn[sset])
            plt.imshow(cross_T[i] - cross_M[i], vmin=ccmin[i], vmax=-ccmin[i])
            cbar = plt.colorbar(orientation="vertical", pad=0.02)
            cbar.set_label(r'$ C^{\xi}_{N}$')
            plt.show()
            
###############################################################################
###############################################################################           
   

     
    # VALIDAZIONE DELLE MATRICI COVARIANZA
    for k in range(3):
        norm_residuals = np.zeros_like(cov_th_0)
        for i in range(Nbins):
            for j in range(Nbins):
                R = cov_T[k][i,j]/(np.sqrt(cov_T[k][i,i]*cov_T[k][j,j]))
                norm_residuals[i,j]=(cov_T[k][i,j]-cov_M[k][i,j])*np.sqrt((Nmeasures-1.)/((1.+R)*cov_T[k][i,i]*cov_T[k][j,j]))
        
        rms_deviation=np.std(norm_residuals)
        
        print("MULTIPOLE "+nm[k]+", measurement set:"+tn[sset])
        print(f"rms deviation of normalized residuals: {rms_deviation}")
        
        if rms_deviation<1.1:
            print("**********")
            print("* PASSED *")
            print("**********")
        else:
            print("!!!!!!!!!!")
            print("! FAILED !")
            print("!!!!!!!!!!")




    # VALIDAZIONE DELLE MATRICI COVARIANZA MISTA
    for k in range(3):
        norm_residuals = np.zeros_like(cov_th_0)
        for i in range(Nbins):
            for j in range(Nbins):
                R = cross_T[k][i,j]/(np.sqrt(cross_T[k][i,i]*cross_T[k][j,j]))
                norm_residuals[i,j]=(cross_T[k][i,j]-cross_M[k][i,j])*np.sqrt((Nmeasures-1.)/((1.+R)*cross_T[k][i,i]*cross_T[k][j,j]))
        
        rms_deviation=np.std(norm_residuals)
        
        print("MULTIPOLE "+cn[k]+", measurement set:"+tn[sset])
        print(f"rms deviation of normalized residuals: {rms_deviation}")
        
        if rms_deviation<1.1:
            print("**********")
            print("* PASSED *")
            print("**********")
        else:
            print("!!!!!!!!!!")
            print("! FAILED !")
            print("!!!!!!!!!!")



    


