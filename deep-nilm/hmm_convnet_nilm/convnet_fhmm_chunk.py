# -*- coding: utf-8 -*-
"""
Created on Friday January 1 20:27:27 2016

@author: Mingjun Zhong, School of Informatics, University of Edinburgh
Email: mingjun.zhong@gmail.com

Requirements:
1) the NILMTK: Non-Intrusive Load Monitoring Toolkit,
        https://github.com/nilmtk/nilmtk;
2) the (free academic) MOSEK license:
        https://www.mosek.com/resources/academic-license

References:
 [1] Mingjun Zhong, Nigel Goddard and Charles Sutton.
 Latent Bayesian melding for integrating individual and population models.
 In Advances in Neural Information Processing Systems 28, 2015.
 [2] Mingjun Zhong, Nigel Goddard and Charles Sutton.
 Signal Aggregate Constraints in Additive Factorial HMMs,
 with Application to Energy Disaggregation.
 In Advances in Neural Information Processing Systems 27, 2014.

Note:
1)  The optimization problems described in the papers were transformed to
    a second order conic programming (SOCP) which suits the MOSEK solver;
    I should write a technical report for this.
2)  Any questions please drop me an email.
"""

# -*- coding: utf-8 -*-

import cPickle
import matplotlib.pyplot as plt
import numpy as np
# from nilmtk.disaggregate.latent_Bayesian_melding import LatentBayesianMelding
# from nilmtk.disaggregate.fhmm_relaxed import FHMM_Relaxed
from fhmm_relaxed import FHMM_Relaxed

############# some global variables ##########################
# Sampling time was 2 minutes
#sample_seconds = 120

####### Select a house: this is the house 2 in UKDALE #########
# This is a demo for applying Latent Bayesian Melding to energy disaggregation.
# The HES data were used for training the model parameters, and the population
# models. This demo shows how to use the LBM to disaggregate the mains readings
# of a day in UKDALE.
# Note that since HES data were read every 2 minutes, so UKDALE data were
# resampled to 2 minutes.
# Note that all the readings were tranferred to the unit: deciwatt-hour,
# which is identical to the unit of HES data.
################################################################

# Note that this demo is only for a chunk (=a day).
# This should be easy to amend for all the data you want to disaggregate.
#########################################################################

## This is the building information: dataset/building_number/date


########### Appliances to be disaggregated: not all of the appliance #########
meterlist = ['kettle', 'dish washer', 'washing machine', 'fridge', 'microwave']
ifile = './optimization_r.pkl'
# meterlist = ['dishwasher',
#             'washingmachine','fridgefreezer']
# meterlist = ['hvac','fridge','dw','dr','wm','light']

## the ground truth reading for those appliances to be disaggregated ########
with open(ifile, 'rb') as infile:
    data = cPickle.load(infile)


## the sum of other meter readings which will not be disaggregated
#groundTruthApplianceReading['othermeters'] = appliancedata.sum(axis=1)

## The mains readings to be disaggregated ###
#mains = meterdata['mains']

#### declare an instance for lbm ################################
# lbm = LatentBayesianMelding()
for part in xrange(0, 50):

    print 'disaggregating: ', part
    index = (part*2000, (part+1)*2000)

    lbm = FHMM_Relaxed(NosOfIters=2, index=index)

# to obtain the model parameters trained by using HES data
    individual_model = lbm.import_model_cPickle(meterlist, data)
# individual_model_lbn = lbm.import_model(meterlist,'lbn.json')

# use lbm to disaggregate mains readings into appliances
# results = lbm.disaggregate_chunk(mains_nipun.astype('float64')[96:192])
    results = lbm.disaggregate_convnet_fhmm_chunk()

# the inferred appliance readings
    infApplianceReading = results['inferred appliance energy']

# compare inferred appliance readings and the ground truth
    for meter in meterlist:
        ifile = './hmm/'+'hmm'+meter+str((part+1)*2000)
        np.save(ifile, infApplianceReading[meter])


# plt.figure()
# plt.plot(data['mains'], color='r', lw=3)
# plt.plot(infApplianceReading['mains'], color='b')
#
# plt.legend(['truth', 'inferred'])
# plt.xlabel('time')
# plt.ylabel('deciwatt-hour')


# close the file
#meterdata_ukdale.close()
