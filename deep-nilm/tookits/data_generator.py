import numpy as np
import pandas as pd
import datetime
import pytz
from readmeters import readdata
from nilmtk import DataSet
import nilmtk.electric as Elec

from time import sleep
import warnings
warnings.filterwarnings("ignore")

def synthetic_data(syn_num, target_name, dataset, target_df, win_length, tar_pro = 0.5, else_pro = 0.3):
    
    house_app = ['washing machine','kettle', 'fridge', 'microwave', 'dish washer']
    house_app.remove(target_name)
    
    data = np.zeros((syn_num,win_length))
    label_data = np.zeros((syn_num,win_length))
    target_num = target_df.shape[0]
    

    
          
    for data_num in xrange(syn_num):

        syn_data = np.zeros((1,win_length))
        
        if np.random.random() < tar_pro:
            
            ran = np.random.randint(0, target_num)
            #print target_df[ran, win_length:2*win_length]
            label = target_df[ran, win_length:2*win_length].reshape(1,win_length).astype('float32')
            syn_data = syn_data + label
            
        else:
            label = np.zeros((1,win_length))
            
        label_data[data_num] = label
            
            
        for app in house_app:
                       
            if np.random.random() < else_pro:
                #print 'adding:', app 
                path = 'E:/dataset/naive_data/House_1_'+app+'_1.npy'
                distrator = np.load(path)
                try:
                    syn_data = syn_data + window_sample(app, dataset, distrator, win_length)
                except:
                    continue
                
        data[data_num] = syn_data
                
    return data, label_data
            
                    
def window_sample(distrator, dataset, distrator_data, win_length, sample_seconds = 6):
    
    data_index = np.random.randint(0, distrator_data.shape[0])
    start = distrator_data[data_index, -2].split('-')
    end = distrator_data[data_index, -1].split('-')
    
    window_st = datetime.datetime(int(start[0]), int(start[1]), int(start[2]), int(start[3]), int(start[4]), int(start[5]))
    window_et = datetime.datetime(int(end[0]), int(end[1]), int(end[2]), int(end[3]), int(end[4]), int(end[5]))
    
    distrator_width = ((window_et - window_st)/6).seconds
    
    window_start = window_st - datetime.timedelta(seconds= win_length * sample_seconds)
    window_end = window_et + datetime.timedelta(seconds= win_length * sample_seconds)
    randn_pad = np.random.randint(0, win_length + distrator_width)
    start_time = window_st + datetime.timedelta(seconds = (randn_pad-win_length)*sample_seconds)
    end_time = window_st + datetime.timedelta(seconds = randn_pad * sample_seconds)
    
    dataset.set_window(start = str(start_time), end = str(end_time))
    elec = dataset.buildings[1].elec
    
    distrator_meter = elec[distrator]
    target_data = distrator_meter.power_series(sample_period=6).next().values
    

    if target_data.shape[0]< win_length:
        tmp = np.zeros((1,win_length))
        tmp[0,0:target_data.shape[0]] = target_data
        target_data = tmp
    return target_data

def readsubmeter(target_app, elec, params, k_dup, sample_seconds, contain_one=1):
# input: 1. target_app
#        2. a panda dataframe elec which contains the mains and submeter readings.
#        3. parameters params: [minoff, minon, onthreshold, windowwidth]
#        4. duplicate each activations for k_dup times
#        5. sample seconds
#        6. whether or not to only allow 1 activation within each window: (default), contain_one=1
# return:1. dataframe duplicated target_ app activations (k*valid activations)*(windowwidth)
#        2. dataframe duplicated mains (which correspond to 1.) (k*valid activations)*(windowwidth)  
        
    mains = elec['mains']
    submeter = elec[target_app]
    activationlist = Elec.get_activations(submeter, min_off_duration = params[0], min_on_duration = params[1], border=1, on_power_threshold = params[2])
    output_mains = np.zeros(shape=(k_dup*len(activationlist),params[3]))
    output_submeter = np.zeros(shape=(k_dup*len(activationlist),params[3]))
    start_end = np.zeros(shape=(k_dup*len(activationlist),2))
    start_end = start_end.astype('string')
    count_dup = 0    
    print 'len(activationlist): ',len(activationlist)
    for (i, actpiece) in enumerate(activationlist):
#         print 'processing activationlist', round(float(i)/len(activationlist),2)*100,'%'
        # compute window limitations
        # make duplications
        if actpiece.index[-1] < (actpiece.index[0]+params[3]-1): # make sure activation fits into the window width
            i_dup = 0
            try_dup = 0
            while i_dup < k_dup and try_dup < 500:
                try_dup += 1
                pad_l = random.randint(0,params[3]+1-len(actpiece))
                pad_r = params[3]-pad_l-len(actpiece)
                if contain_one==1: # if only allow 1 activation within a window
                    if i>0:
                        last_end = activationlist[i-1].index[-1]
                    else:
                        last_end = submeter.index[0]
                    if i<(len(activationlist)-1):
                        next_start = activationlist[i+1].index[0]
                    else:
                        next_start = submeter.index[-1]
                    try:
                        nan_check = (not np.isnan(submeter[actpiece.index[0]-pad_l:actpiece.index[-1]+pad_r].values.reshape(1,params[3])).any()) and (not np.isnan(mains[actpiece.index[0]-pad_l:actpiece.index[-1]+pad_r].values.reshape(1,params[3])).any())
                        dup_check = duplicated_check(actpiece.index[0]-pad_l, actpiece.index[-1]+pad_r, last_end, next_start)
#                         print 'nan_check', nan_check, 'dup_check',dup_check
                        if nan_check and dup_check:
                            output_mains[count_dup:count_dup+1] = mains[actpiece.index[0]-pad_l:actpiece.index[-1]+pad_r].values.reshape(1,params[3])                       
                            output_submeter[count_dup:count_dup+1] = submeter[actpiece.index[0]-pad_l:actpiece.index[-1]+pad_r].values.reshape(1,params[3])
                            start_end[count_dup,0] = actpiece.index[0].strftime('%Y-%m-%d-%H-%M-%S')
                            start_end[count_dup,1] = actpiece.index[-1].strftime('%Y-%m-%d-%H-%M-%S')
                            count_dup = count_dup + 1
                            i_dup += 1
#                             print 'samples generated: ', count_dup
                        else:
                            1
#                             print 'FAIL THE CHECK'
                    except:
                        1
                        print 'CAN NOT CHECK'
                else:
                    try:
                        output_mains[count_dup:count_dup+1] = mains[actpiece.index[0]-pad_l:actpiece.index[-1]+pad_r].values.reshape(1,params[3])
                        output_submeter[count_dup:count_dup+1] = submeter[actpiece.index[0]-pad_l:actpiece.index[-1]+pad_r].values.reshape(1,params[3])
                        start_end[count_dup,0] = actpiece.index[0]
                        start_end[count_dup,1] = actpiece.index[-1]
                        count_dup = count_dup + 1
                        i_dup += 1
                    except:
                        1
    # # # delete blank rows (caused by fail the duplicated_check)
    print 'stacking and deleting..'
    output = np.hstack([output_mains, output_submeter, start_end])
    output = np.delete(output ,range(count_dup,k_dup*len(activationlist)),axis=0)
#     print 'count_dup',count_dup
    return output
 
    
def data_stacker(data, width, stepsize=1):
    return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width))
