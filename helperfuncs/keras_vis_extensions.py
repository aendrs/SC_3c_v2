# -*- coding: utf-8 -*-
"""
Extension to keras-vis to support multiplpe inputs

@author: andre
"""

from vis.losses import ActivationMaximization
#from vis.visualization import visualize_activation_with_losses
from vis.visualization import get_num_filters
from vis.regularizers import TotalVariation, LPNorm
from vis.optimizer import Optimizer
from vis.backprop_modifiers import get
from vis.utils import utils
from vis.input_modifiers import Jitter

from keras import backend as K
import numpy as np







def visualize_filters_doublepass(model, layer_idx, input_idx, #see info above about input_idx
                         input_range=(0, 255), titlestr='Filter',
                         filters='all',
                         act_max_weight_1=1, lp_norm_weight_1=10, p_1=6, tv_weight_1=0,
                         max_iter_1=300, input_modifiers_1=[Jitter(1)],
                         act_max_weight_2=1, lp_norm_weight_2=10, p_2=6, tv_weight_2=.05,
                         max_iter_2=300, input_modifiers_2=[Jitter(1)]):
    ''' 
    Create images of the filter activations in a layer. ALL OF THEM
    Most of the arguments are the same as the ones for visualize_activation
    '''
    # index all filters in this layer. (or use the provided indices)
    if filters =='all':
        filters = np.arange(get_num_filters(model.layers[layer_idx]))
    else:
        filters=filters
        
    vis_images = []
    for idx in filters:
        print('processing Filter {}, first pass'.format(idx))
        img = visualize_activation_multiinputs(model=model,layer_idx=layer_idx, 
                                               input_idx=input_idx, 
                                               filter_indices=idx,
                                               input_range=input_range,
                                               act_max_weight= act_max_weight_1,
                                               lp_norm_weight=lp_norm_weight_1,
                                               p=p_1,
                                               tv_weight=tv_weight_1,
                                               input_modifiers=input_modifiers_1,
                                               max_iter=max_iter_1)
        vis_images.append(img)

    new_vis_images = []
    for i, idx in enumerate(filters):
        # We will seed with optimized image this time.
        print('processing Filter {}, second pass'.format(idx))
        img = visualize_activation_multiinputs(model, layer_idx=layer_idx, 
                                               input_idx=input_idx,
                                               filter_indices=idx, 
                                               seed_input=vis_images[i],
                                               act_max_weight=act_max_weight_2,
                                               lp_norm_weight=lp_norm_weight_2,
                                               p=p_2,
                                               tv_weight=tv_weight_2,
                                               input_modifiers=input_modifiers_2,
                                               max_iter=max_iter_2)
    
        # Utility to overlay text on image.
        img = utils.draw_text(img,(titlestr + ' {}'.format(idx)))    
        new_vis_images.append(img)
    return (new_vis_images,vis_images)







def visualize_activation_multiinputs(model, layer_idx, input_idx=0, filter_indices=None,
                         seed_input=None, input_range=(0, 255),
                         backprop_modifier=None, grad_modifier=None,
                         act_max_weight=1, lp_norm_weight=10, p=6, tv_weight=10,
                         **optimizer_params):
    """
    Modified version of visualize_activation from keras-vis package (january 2018)
    Added the option to specify the index of a multiple input model

    input_idx:  index of the multiple input for the model, check model.input to see the list structure of the inputs 
    
    Read Original Documentation of keras-vis.visualize_activation!!!!!!!!
    """

    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)

    if type(model.input) is not list: #check if model.input has a single input or is multiinput
        input=model.input
    else:
        input=model.input[input_idx]

    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), act_max_weight),
        (LPNorm(input,p=p), lp_norm_weight),
        (TotalVariation(input), tv_weight)
    ]

    # Add grad_filter to optimizer_params.
    optimizer_params = utils.add_defaults_to_kwargs({
        'grad_modifier': grad_modifier
    }, **optimizer_params)

    return visualize_activation_with_losses_multiinput(input, losses, seed_input, input_range, **optimizer_params)



def visualize_activation_with_losses_multiinput(input_tensor, losses,
                                     seed_input=None, input_range=(0, 255),
                                     **optimizer_params):
    """Generates the `input_tensor` that minimizes the weighted `losses`. This function is intended for advanced
    use cases where a custom loss is desired.

    Args:
        input_tensor: An input tensor of shape: `(samples, channels, image_dims...)` if `image_data_format=
            channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        seed_input: Seeds the optimization with a starting image. Initialized with a random value when set to None.
            (Default value = None)
        input_range: Specifies the input range as a `(min, max)` tuple. This is used to rescale the
            final optimized input to the given range. (Default value=(0, 255))
        optimizer_params: The **kwargs for optimizer [params](vis.optimizer#optimizerminimize). Will default to
            reasonable values when required keys are not found.

    Returns:
        The model input that minimizes the weighted `losses`.
    """
    # Default optimizer kwargs.
    optimizer_params = utils.add_defaults_to_kwargs({
        'seed_input': seed_input,
        'max_iter': 200,
        'verbose': False
    }, **optimizer_params)

    opt = Optimizer(input_tensor, losses, input_range)
    img = opt.minimize(**optimizer_params)[0]

    # If range has integer numbers, cast to 'uint8'
    if isinstance(input_range[0], int) and isinstance(input_range[1], int):
        img = np.clip(img, input_range[0], input_range[1]).astype('uint8')
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #img=((img-img.min())/(img.max()-img.min()))*255
        #img=img.astype('uint8')
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    if K.image_data_format() == 'channels_first':
        img = np.moveaxis(img, 0, -1)
    return img


###==========================================================================
###==========================================================================
###==========================================================================
###==========================================================================



def visualize_activation_MOD (model, layer_idx, input_idx=0, filter_indices=None,
                         seed_input=None, input_range=(0, 255),
                         backprop_modifier=None, grad_modifier=None,
                         act_max_weight=1, lp_norm_weight=10, p=6, tv_weight=10,
                         **optimizer_params):

    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)

    if type(model.input) is not list: #check if model.input has a single input or is multiinput
        model_input=model.input
        model_type='single_input'
    else:
        # maybe pass the whole input to the optimizer class??? and inside check if its multiple or what after the calculation of grads
        model_input=model.input
        model_type='multi_input'

    ######## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
   
    #calculate an initial gradient to determine the dependencies of the output wrt the input. 
    #If the model has a single input or if the uoutput depends on fewer tensors than the total number of inputs, some of the gradients will be None
    protoloss=ActivationMaximization(model.layers[layer_idx], 0)
    protograd=K.gradients(protoloss.build_loss(), model.input)
    
    
    losses=[]
    if model_type=='single_input':
        losses = [(ActivationMaximization(model.layers[layer_idx], filter_indices), act_max_weight),
        (LPNorm(model_input, p=p), lp_norm_weight),
        (TotalVariation(model_input), tv_weight)]   
    else:
        losses.append((ActivationMaximization(model.layers[layer_idx], filter_indices), act_max_weight))
        for i,grad in enumerate(protograd):
            if grad==None:
                continue
            else:
                losses.extend([(LPNorm(model.input[i], p=p), lp_norm_weight),(TotalVariation(model.input[i]), tv_weight)])
  
    
    '''    
        losses = [(ActivationMaximization(model.layers[layer_idx], 1), 1),
        (LPNorm(model.input[0], p=p), 10),
        (TotalVariation(model.input[0]), 10),
        (LPNorm(model.input[1], p=p), 10),
        (TotalVariation(model.input[1]), 10),
        (LPNorm(model.input[2], p=p), 10),
        (TotalVariation(model.input[2]), 10)] 
    
    (ActivationMaximization(model.layers[layer_idx], filter_indices), act_max_weight),
        (LPNorm(regularizer_input, p=p), lp_norm_weight),
        (TotalVariation(regularizer_input), tv_weight) 
        
        
    
    losses.extend([(LPNorm(model.input[0], p=p), 10),(TotalVariation(model.input[0]), 10)])
    
    losses = [
        (ActivationMaximization(model.layers[layer_idx], 1), 1),
        (LPNorm(model.input[0], p=p), 10),
        (TotalVariation(model.input[0]), 10),
        (LPNorm(model.input[1], p=p), 10),
        (TotalVariation(model.input[1]), 10),
        (LPNorm(model.input[2], p=p), 10),
        (TotalVariation(model.input[2]), 10)] 
    '''

    # Add grad_filter to optimizer_params.
    optimizer_params = utils.add_defaults_to_kwargs({
        'grad_modifier': grad_modifier
    }, **optimizer_params)


    # Default optimizer kwargs.
    optimizer_params = utils.add_defaults_to_kwargs({'seed_input': seed_input,'max_iter': 200,'verbose': False}, **optimizer_params)
    opt = Optimizer_MOD(model_input, input_idx, model_type, losses, input_range)
    #img = opt.minimize(**optimizer_params)[0]
    #img=opt.minimize(seed_input=seed_input, max_iter=200)
    img = opt.minimize(**optimizer_params)[0]
    
    # !!!!!!!!!!!!!!!!!!!!!! Check all this deal of converting to uint8 and the input and output ranges
    # If range has integer numbers, cast to 'uint8'
    if isinstance(input_range[0], int) and isinstance(input_range[1], int):
        if model_type=='single_input':
            img = np.clip(img, input_range[0], input_range[1]).astype('uint8')
        else:
            for i in range(len(img)):
                img[i]=np.clip(img[i],input_range[0],input_range[1]).astype('uint8')
      
    if K.image_data_format() == 'channels_first':
        img = np.moveaxis(img, 0, -1)
    return img










#===============================================================================
## Create a modified OPTIMIZER class to fit multi-input cases, for reference check the original class from keras-vis
    

from vis.callbacks import Print
from vis.grad_modifiers import get as get_gradmod



_PRINT_CALLBACK = Print()


def _identity(x):
    return x


class Optimizer_MOD(object):

    def __init__(self, model_input, input_idx, model_type, 
                 losses, input_range=(0, 255), wrt_tensor=None, norm_grads=True):

        self.input_tensor = model_input
        self.input_idx=input_idx
        self.input_range = input_range
        self.model_type=model_type
        self.loss_names = []
        self.loss_functions = []
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        self.wrt_tensor = self.input_tensor if wrt_tensor is None else wrt_tensor

        overall_loss = None
        for loss, weight in losses:
            # Perf optimization. Don't build loss function with 0 weight.
            if weight != 0:
                loss_fn = weight * loss.build_loss()
                overall_loss = loss_fn if overall_loss is None else overall_loss + loss_fn
                self.loss_names.append(loss.name)
                self.loss_functions.append(loss_fn)


        # If 'single input' then only one grad is computed, else all of them (however some may be NONE, take care of that)
        if self.model_type=='single_input':
            # Compute gradient of overall with respect to `wrt` tensor.
            grads = K.gradients(overall_loss, self.wrt_tensor)[0]
        elif self.model_type=='multi_input':
            # compute gradients and then check if the loss depends on all inputs or a subset
            grads = K.gradients(overall_loss, self.wrt_tensor)
            if None in grads:
                #grads=grads[self.input_idx]               
                grads=[grads[i] for i in range(len(grads)) if grads[i] is not None]
                
            # ^ if None in grads, use the input defined (which is not none), ELSE use all the inputs, 
            # this happens when a CNN layer depends on just a single input, before concatenation
            
        #important to know if all branches are used in multi-input models, used in minimize method    
        self.grads_length=1 if type(grads) is not list else len(grads)
        
        
        
        #Normalize gradients
        if norm_grads:
            if type(grads) is not list: #in the case that there is just one gradient
                grads = grads / (K.sqrt(K.mean(K.square(grads))) + K.epsilon())
            else:
                normgrads=[]
                for i,tensor in enumerate(grads):      
                    temp = tensor / (K.sqrt(K.mean(K.square(tensor))) + K.epsilon())
                    normgrads.append(temp)    
                grads=normgrads
 

        # (original )The main function to compute various quantities in optimization loop.
        #self.compute_fn = K.function([self.input_tensor, K.learning_phase()],
        #                             self.loss_functions + [overall_loss, grads, self.wrt_tensor])

        grads_opt=[]
        grads_opt.extend([overall_loss])
        if type(grads) is list:
            grads_opt.extend(grads)        
        else:
            grads_opt.append(grads)


        if type(self.wrt_tensor) is list:
            grads_opt.extend(self.wrt_tensor)
            print('list extend')
        else:
            grads_opt.append(self.wrt_tensor)
            print('list append')

        # The main function to compute various quantities in optimization loop.
        # !!!!!!
        #self.compute_fn = K.function([self.input_tensor, K.learning_phase()],
        #                             self.loss_functions + grads_opt)
        
        if self.model_type=='single_input':
            self.compute_fn = K.function([self.input_tensor, K.learning_phase()],
                                     self.loss_functions + grads_opt)
        else:
            self.compute_fn = K.function( self.input_tensor + [K.learning_phase()],
                                     self.loss_functions + grads_opt)

        
    def _rmsprop(self, grads, cache=None, decay_rate=0.95):
    
        """Uses RMSProp to compute step from gradients.

        Args:
            grads: numpy array of gradients.
            cache: numpy array of same shape as `grads` as RMSProp cache
            decay_rate: How fast to decay cache

        Returns:
            A tuple of
                step: numpy array of the same shape as `grads` giving the step.
                    Note that this does not yet take the learning rate into account.
                cache: Updated RMSProp cache.
        """
        if cache is None:
            cache = np.zeros_like(grads)
        cache = decay_rate * cache + (1 - decay_rate) * grads ** 2
        step = -grads / np.sqrt(cache + K.epsilon())
        return step, cache

    def _get_seed_input(self, seed_input):
        """Creates a random `seed_input` if None. Otherwise:
            - Ensures batch_size dim on provided `seed_input`.
            - Shuffle axis according to expected `image_data_format`.
        """
        desired_shape = (1, ) + K.int_shape(self.input_tensor[0])[-3:] # Does it cause a problem with single input models? *update,(with the [-3:] it might work great for both cases)
        if seed_input is None:
            return utils.random_array(desired_shape, mean=np.mean(self.input_range),
                                      std=0.05 * (self.input_range[1] - self.input_range[0]))

        # Add batch dim if needed.
        if len(seed_input.shape) != len(desired_shape):
            seed_input = np.expand_dims(seed_input, 0)

        # Only possible if channel idx is out of place.
        if seed_input.shape != desired_shape:
            seed_input = np.moveaxis(seed_input, -1, 1)
        return seed_input.astype(K.floatx())

    def minimize(self, seed_input=None, max_iter=200,
                 input_modifiers=None, grad_modifier=None,
                 callbacks=None, verbose=True):
        """Performs gradient descent on the input image with respect to defined losses.
        
        Args:
            seed_input: 
           ***     *NEW* a numpy array (for single_input models) or a list of arrays (for multi_input models) **********
                An N-dim numpy array of shape: `(samples, channels, image_dims...)` if `image_data_format=
                channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
                Seeded with random noise if set to None. (Default value = None)
            max_iter: The maximum number of gradient descent iterations. (Default value = 200)
            input_modifiers: A list of [InputModifier](vis.input_modifiers#inputmodifier) instances specifying
                how to make `pre` and `post` changes to the optimized input during the optimization process.
                `pre` is applied in list order while `post` is applied in reverse order. For example,
                `input_modifiers = [f, g]` means that `pre_input = g(f(inp))` and `post_input = f(g(inp))`
            grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
                specify anything, gradients are unchanged. (Default value = None)
            callbacks: A list of [OptimizerCallback](vis.callbacks#optimizercallback) instances to trigger.
            verbose: Logs individual losses at the end of every gradient descent iteration.
                Very useful to estimate loss weight factor(s). (Default value = True)

        Returns:
            The tuple of `(optimized input, grads with respect to wrt, wrt_value)` after gradient descent iterations.
            
            ##seed_input.append(self._get_seed_input(seed_input[i] if len(seed_input)>0 else None))
        """
        
        if self.model_type=='single_input':
            seed_input = self._get_seed_input(seed_input)
        else:
            temp_list=[]
            for i in range(len(self.input_tensor)):
                if seed_input==None:
                    temp_list.append(self._get_seed_input(None))
                else:
                    temp_list.append(self._get_seed_input(seed_input[i]))
            seed_input=temp_list
            
        
        input_modifiers = input_modifiers or []
        
        # !!! TODO : adapt get_gradmod for multi input models
        grad_modifier = _identity if grad_modifier is None else get_gradmod(grad_modifier)
        
        # !!! TODO : adapt callbacks for multi input models
        callbacks = callbacks or []
        if verbose:
            callbacks.append(_PRINT_CALLBACK)
            
        if self.grads_length ==1:    
            cache = None
            step = None
        else:
            cache=[None]*self.grads_length #make it an empty list of the right shape
            step=[None]*self.grads_length
            
        best_loss = float('inf')
        best_input = None

        grads = None 
        wrt_value = None

        for i in range(max_iter):
            # Apply modifiers `pre` step
            for modifier in input_modifiers:
                if self.model_type=='single_input':
                    seed_input = modifier.pre(seed_input)
                else:
                    for i in range(len(self.input_tensor)):
                        seed_input[i]=modifier.pre(seed_input[i])
            
            
            # Main keras function            
            # 0 learning phase for 'test'
            #----------------------------------------------------------------
            if self.model_type=='single_input':
                computed_values = self.compute_fn([seed_input, 0])
            else:
                computed_values = self.compute_fn(seed_input +[0])
            #----------------------------------------------------------------
            
            losses = computed_values[:len(self.loss_names)]
            named_losses = zip(self.loss_names, losses)
            #overall_loss, grads, wrt_value = computed_values[len(self.loss_names):]

            overall_loss = computed_values[len(self.loss_names)]
            grads= computed_values[len(self.loss_names)+1:len(self.loss_names)+1+self.grads_length] #take into account the grads number which is not always the same as the number of inputs
            if self.model_type=='single_input':
                wrt_value = computed_values[-1] #tensor
            else:
                wrt_value = computed_values[-len(self.input_tensor):] #list of tensors

            #theano grads shape is inconsistent for some reason. Patch for now and investigate later.
            #if grads.shape != wrt_value.shape:
            #    grads = np.reshape(grads, wrt_value.shape)

            # !!! TODO : adapt for multi input ================================
            # Apply grad modifier.
            grads = grad_modifier(grads)

            # Trigger callbacks
            for c in callbacks:
                c.callback(i, named_losses, overall_loss, grads, wrt_value)

            # Gradient descent update.
            # It only makes sense to do this if wrt_tensor is input_tensor. Otherwise shapes wont match for the update.
            
            if self.wrt_tensor is self.input_tensor:
                if self.grads_length ==1: #if the output depends on only one input (either single-input model or just a filter in a single branch)
                    step, cache = self._rmsprop(grads[0], cache)
                    if self.model_type=='single_input':
                        seed_input += step
                    else:
                        for i in range(len(seed_input)):
                            seed_input[i]=seed_input[i]+step #it copies the step to every input, although just one is necessary, it is redundant and may need to be changed later but I dont believe it causes a problem since we are only interested in a single input tensor
                else:
                    for i, grad_X in enumerate(grads):
                        step[i], cache[i] = self._rmsprop(grad_X,cache[i])
                        seed_input[i] += step[i]


            # Apply modifiers `post` step
            for modifier in reversed(input_modifiers):
                if self.model_type=='single_input':
                    seed_input = modifier.post(seed_input)
                else:
                     for i in range(len(self.input_tensor)):
                        seed_input[i]=modifier.post(seed_input[i])

            if overall_loss < best_loss:
                best_loss = overall_loss.copy()
                best_input = seed_input.copy()

        # Trigger on_end
        for c in callbacks:
            c.on_end()
        
        output_img=[]
        
        if self.model_type=='single_input':
            output_img=utils.deprocess_input(best_input[0], self.input_range)
        else:
            for i in range(len(best_input)):
                output_img.append(utils.deprocess_input(best_input[i][0],self.input_range))

        return output_img, grads, wrt_value







