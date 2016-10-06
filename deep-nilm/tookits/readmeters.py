import pandas as pd
import pdb 

def readdata(metergroup, sample_seconds, submeters = 'all'):  
    """
    return the dataframe from uk-dale
    
    parameters:
    ---------------------------------------------------------
    metergroup: data groups
    sample_seconds: int, the sample seconds for data
    submeters: list, required submeter list
    
    """    
    # Use mains to extract the time stamp as the index
    mains = metergroup.mains().power_series(sample_period=sample_seconds).next()
    index = mains.index
    #print 'index: ',index
    # This is to show the appliance and submeter time series
    check_mains = mains.empty
    if submeters == 'all':
        
        for (i, meter) in enumerate(metergroup.submeters().meters):
            meter_data = meter.power_series(sample_period=sample_seconds).next()      

            if meter_data.empty:
                print("Meter: {} is empty".format(meter.appliances[0].label()))
            else:
                if i == 0:
                    meterdata = pd.concat([pd.DataFrame(index=index),meter_data],
                                               ignore_index=True,axis=1).fillna(0) 
                    print meter.appliances[0].label()
                    meterdata.columns = [meter.appliances[0].label()]
                else:
                    try:
                        meterdata[meter.appliances[0].label()] = \
                                    pd.DataFrame(meter_data,index=index).fillna(0)
                    except ValueError:
                        print("Filling missing values for appliance: {}".format(meter.appliances[0].label()))                    
                        df = pd.concat([pd.DataFrame(index=index),meter_data],
                                           ignore_index=True, axis=1).fillna(0)
                        meterdata[meter.appliances[0].label()] = df
                        
                meterdata['mains'] = df = pd.concat([pd.DataFrame(index=index),mains],
                                               ignore_index=True, axis=1).fillna(0)
                            
    else: 
        assert type(submeters) == list 
        success_num = 0
        for (i, submeter) in enumerate(submeters): 
            
            label = submeter + "_1"
            try:
                meter=metergroup[submeter] 
            except:
                continue
            success_num += 1
            meter_data = meter.power_series(sample_period=sample_seconds).next()      

            if meter_data.empty:
                print("Meter: {} is empty".format(meter.appliances[0].label()))
                
            else:
                if success_num <= 1:
                    meterdata = pd.concat([pd.DataFrame(index=index),meter_data],
                                               ignore_index=True,axis=1).fillna(0)
                    meterdata.columns = [label]
                    #print meter.appliances[0].label()
                            #meterdata[meter.appliances[0].label()] = \
                            #            meter_data.values.reshape(len(index), 1)
                else:
                    try:
                        meterdata[label] = \
                                pd.DataFrame(meter_data,index=index).fillna(0)
                                        #meter_data.values.reshape(len(meter_data.index), 1)
                    except:
                        meterdata = pd.concat([pd.DataFrame(index=index),meter_data],
                                               ignore_index=True,axis=1).fillna(0)
                        meterdata.columns = [label]
                        try: 
                            meterdata[label] = \
                                pd.DataFrame(meter_data,index=index).fillna(0)
                        except:
                            print("Filling missing values for appliance: {}".format(meter.appliances[0].label()))                    
                            df = pd.concat([pd.DataFrame(index=index),meter_data],
                                               ignore_index=True, axis=1).fillna(0)
                            meterdata[meter.appliances[0].label()] = df
                            
                meterdata['mains'] = pd.concat([pd.DataFrame(index=index),mains],
                                       ignore_index=True, axis=1).fillna(0)
    try:                
        return meterdata
    except:
        meterdata = pd.DataFrame()
        return meterdata

def truncate_meter(meter,threshold):
    # This method is to truncate meters. The meter reading is set to zero when
    # it is smaller than threshold
    meter_clip = meter.clip(lower = threshold)
    meter_chop = meter_clip.replace(to_replace=threshold,value=0.0)
    return meter_chop

def duplicated_check(win_start, win_stop, l_end, n_start):
    
    if (win_start > l_end) and (win_stop < n_start):    
        return True
    
    return False
