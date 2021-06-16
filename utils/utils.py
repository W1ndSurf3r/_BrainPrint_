import torch as th
from braindecode.datautil.splitters import split_into_two_sets, concatenate_sets
from braindecode.datautil.signal_target import SignalAndTarget



def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x
    
def square(x):
    return x * x

def no_square(x):
    return x

def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return th.log(th.clamp(x, min=eps))


def identity(x):
    return x    




def windows(data, size, step):
	start = 0
	while ((start+size) < data.shape[0]):
		yield int(start), int(start + size)
		start += step


def segment_signal_without_transition(data, window_size, step):
	segments = []
	for (start, end) in windows(data, window_size, step):
		if(len(data[start:end]) == window_size):
			segments = segments + [data[start:end]]
	return np.array(segments)


def segment_dataset(X, window_size, step):
	win_x = []
	for i in range(X.shape[0]):
		win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
	win_x = np.array(win_x)
	return win_x




"""________________________________________________________________________________________________________________________"""
import numpy as np
def augment_signals(data, augment_by_samples= 100, seconds_to_swap=[0,1]):
    
    x = data.X.copy()
    y = data.y.copy()
    print("first second: {}:{}, second: {}:{}, shape: {}".format(seconds_to_swap[0]*250 , seconds_to_swap[0]+ augment_by_samples,  seconds_to_swap[1]*250,seconds_to_swap[1]*250+ augment_by_samples,x.shape))
    # assert (seconds_to_swap[0]*250 <= x.shape[3] and seconds_to_swap[1]*250 <= x.shape[3]), "Seconds don't exist, check your input"
    new_arr_x = x.copy()
    new_arr_y = y.copy()
    # print('DATA: ', x.shape)
    for i, (trial, label) in enumerate(zip(x,y)):
        
        first_crop = int(seconds_to_swap[0]*250) #250 should be fs later
        second_crop =  int(seconds_to_swap[1]*250)
        # print(first_crop, second_crop)
        # print(trial.shape)
        temp = trial[:,first_crop: first_crop+ augment_by_samples]
        #print('first crop: ',first_crop, first_crop+ augment_by_samples, temp.shape)
        temp2 = trial[:,second_crop: second_crop + augment_by_samples]
        #print('first crop: ',second_crop, second_crop+ augment_by_samples, temp2.shape)
        #np.testing.assert_array_equal(temp,temp2)
        #print(temp2.shape, temp.shape)
        new_arr_x[i,:,first_crop: first_crop+ augment_by_samples] = temp2
        new_arr_x[i,:, second_crop: second_crop + augment_by_samples] = temp
        new_arr_y[i] = label
    new_data = SignalAndTarget(new_arr_x, new_arr_y)
    return new_data

def conc_augmented(train):
    #print(train.X.shape, 'adsadasdas')
    augmented_train_1 = augment_signals(train,seconds_to_swap=[0,0.4]) 
    augmented_train_2 = train
    train = concatenate_sets([augmented_train_1,augmented_train_2])
        
    return train


class MaxNormDefaultConstraint(object):
    """
    Applies max L2 norm 2 to the weights until the final layer and L2 norm 0.5
    to the weights of the final layer as done in [1]_.
    
    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., 
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
   Deep learning with convolutional neural networks for EEG decoding and
   visualization.
   Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730

"""

    def apply(self, model):
        last_weight = None
        for name, module in list(model.named_modules()):
            if hasattr(module, "weight") and (
                not module.__class__.__name__.startswith("BatchNorm")
            ):
                module.weight.data = torch.renorm(
                    module.weight.data, 2, 0, maxnorm=2
                )
                last_weight = module.weight
                #print(name)
                #print('applying constraints')
        if last_weight is not None:
            last_weight.data = torch.renorm(last_weight.data, 2, 0, maxnorm=0.5)

def _get_padding(padding_type, kernel_size):
    #assert isinstance(kernel_size, int)
    assert padding_type in ['SAME', 'VALID'] 
    if padding_type == 'SAME':
        return (kernel_size - 1) // 2

    return 0

def _calculate_output(H,padding,dilation, kernel_size, stride):

    numerator = (H + 2*padding-dilation * (kernel_size -1) - 1 )
    denominator = stride
    H_out = (numerator/denominator) + 1

    return H_out

def _calculate_strided_padding(W, F, S):
     #  W= Input Size , F = filter size (kernel), S = stride,
    P = ((S-1)*W-S+F)//2

    return P

def get_dilated_kernel(k,d):
    #kernel size, dilation
    new_k = k+(k-1)*(d-1)

    return new_k


def crop(train_set, test_set):
    window_size = 200
    step = 50
    print('Window size: {}, step: {}'.format(window_size, step))
    #n_channel = 3

    test_X	= test_set.X # [trials, channels, time length]
    train_X	= train_set.X

    test_y	= test_set.y.ravel()
    train_y = train_set.y.ravel()


    #train_y = np.asarray(pd.get_dummies(train_y), dtype = np.int8)
    #test_y = np.asarray(pd.get_dummies(test_y), dtype = np.int8)
    train_raw_x = np.transpose(train_X, [0, 2, 1])
    test_raw_x = np.transpose(test_X, [0, 2, 1])


    train_win_x = segment_dataset(train_raw_x, window_size, step)
    print("train_win_x shape: ", train_win_x.shape)
    test_win_x = segment_dataset(test_raw_x, window_size, step)
    print("test_win_x shape: ", test_win_x.shape)

    # [trial, window, channel, time_length]
    train_win_x = np.transpose(train_win_x, [0, 1, 3, 2])
    print("train_win_x shape: ", train_win_x.shape)

    test_win_x = np.transpose(test_win_x, [0, 1, 3, 2])
    print("test_win_x shape: ", test_win_x.shape)


    # [trial, window, channel, time_length, 1]
    train_x = np.expand_dims(train_win_x, axis = 4)
    test_x = np.expand_dims(test_win_x, axis = 4)
    train_set.X = train_x
    #train_set.y = train_y
    
    test_set.X = test_x
    #test_set.y = test_y
    return train_set,test_set
