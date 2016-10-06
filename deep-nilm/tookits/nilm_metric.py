import numpy as np
from sklearn.metrics import confusion_matrix

def get_TP(target, prediction, threshold):
    '''
    compute the  number of true positive
    
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float    
    '''

    assert (target.shape == prediction.shape)
    
    target = 1-np.clip(target, threshold, 0)/threshold
    prediction = 1-np.clip(prediction, threshold, 0)/threshold
    
    tp_array = np.logical_and(target,prediction)*1.0
    tp = np.sum(tp_array)
    
    return tp

def get_FP(target, prediction, threshold):
    '''
    compute the  number of false positive
    
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float    
    '''
    
    assert (target.shape == prediction.shape)
    
    target = np.clip(target, threshold, 0)/threshold
    prediction = 1-np.clip(prediction, threshold, 0)/threshold
    
    fp_array = np.logical_and(target,prediction)*1.0
    fp = np.sum(fp_array)
    
    return fp

def get_FN(target, prediction, threshold):
    '''
    compute the  number of false negtive
    
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float    
    '''
    
    assert (target.shape == prediction.shape)
    
    target = 1-np.clip(target, threshold, 0)/threshold
    prediction = np.clip(prediction, threshold, 0)/threshold
    
    fn_array = np.logical_and(target,prediction)*1.0
    fn = np.sum(fn_array)
    
    return fn


def get_TN(target, prediction, threshold):
    
    '''
    compute the  number of true negative
    
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float    
    '''
    
    assert (target.shape == prediction.shape)
    
    target = np.clip(target, threshold, 0)/threshold
    prediction = np.clip(prediction, threshold, 0)/threshold
    
    tn_array = np.logical_and(target,prediction)*1.0
    tn = np.sum(tn_array)
    
    return tn

def get_recall(target, prediction, threshold):
    '''
    compute the recall rate
    
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float    
    '''
    
    tp = get_TP(target, prediction, threshold)
    fn = get_FN(target, prediction, threshold)
    
    recall = tp/(tp+fn)
    return recall

def get_precision(target, prediction, threshold):
    
    '''
    compute the  precision rate
    
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float    
    '''
    
    tp = get_TP(target, prediction, threshold)
    fp = get_FP(target, prediction, threshold)
    
    precision = tp/(tp+fp)
    return precision

def get_F1(target, prediction, threshold):
    
    '''
    compute the  F1 score
    
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float    
    ''' 
   
    recall = get_recall(target, prediction, threshold)
    precision = get_precision(target, prediction, threshold)
    f1 = 2*precision*recall/(precision+recall)
    return f1

def get_accuracy(target, prediction, threshold):
    
    '''
    compute the accuracy rate
    
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float    
    '''    

    
    tp = get_TP(target, prediction, threshold)
    tn = get_TN(target, prediction, threshold)
    
    accuracy = (tp+tn)/target.size
    
    return accuracy

def get_relative_error(target, prediction):    
        
    '''
    compute the  relative_error
    
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array  
    '''
    
    assert (target.shape == prediction.shape)
    
    return np.mean(np.nan_to_num(np.abs(target-prediction)/np.maximum(target,prediction)))

def get_abs_error(target, prediction):
    
    '''
    compute the  absolute_error
    
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array  
    '''
    
    
    assert (target.shape == prediction.shape)
    
    return np.mean(np.abs(target-prediction))

def get_nde(target, prediction):
    
    '''
    compute the  normalized disaggregation error
    
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array  
    '''
    
    return np.sum((target-prediction)**2)/np.sum((target**2))


def get_confusion_accuracy(target, prediction, class_num):
    '''
    compute the  confusion_accuracy
    
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array  
    '''    
    
    confusion = confusion_matrix(target, prediction).diagonal()
    confusion_count = np.zeros((class_num))
    
    unique, counts = np.unique(target, return_counts=True)
    prediction_unique = np.unique(prediction, return_counts=False)
    union = np.union1d(unique,  prediction_unique).astype('int64')
    confusion_count[union] = confusion
    
    state_count = np.zeros((class_num))
    unique = unique.astype('int64')
    state_count [unique] = counts
    
    accu_vec = confusion_count/state_count 
    accu_vec = accu_vec[~np.isnan(accu_vec)]
    
    return accu_vec.mean()
    

