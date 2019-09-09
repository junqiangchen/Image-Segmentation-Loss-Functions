def categorical_crossentropy(Y_pred, Y_gt):
    """
    Categorical crossentropy between an output and a target
    loss=-y*log(y')
    :param Y_pred: A tensor resulting from a softmax
    :param Y_gt:  A tensor of the same shape as `output`
    :return:categorical_crossentropy loss
    """
    epsilon = 1.e-5
    # scale preds so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keep_dims=True)
    # manual computation of crossentropy
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    loss = -Y_gt * tf.log(output)
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_mean(loss)
    return loss
	
	
def weighted_categorical_crossentropy(Y_pred, Y_gt, weights):
    """
    weighted_categorical_crossentropy between an output and a target
    loss=-weight*y*log(y')
    :param Y_pred:A tensor resulting from a softmax
    :param Y_gt:A tensor of the same shape as `output`
    :param weights:numpy array of shape (C,) where C is the number of classes
    :return:categorical_crossentropy loss
    Usage:
    weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
    """
    weights = np.array(weights)
    epsilon = 1.e-5
    # scale preds so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keep_dims=True)
    # manual computation of crossentropy
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    loss = - Y_gt * tf.log(output)
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_mean(weights * loss)
    return loss
	
	
def categorical_dice(Y_pred, Y_gt, weight_loss):
    """
    multi label dice loss with weighted
    WDL=1-2*(sum(w*sum(r&p))/sum((w*sum(r+p)))),w=array of shape (C,)
    :param Y_pred: [None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_pred is softmax result
    :param Y_gt:[None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_gt is one hot result
    :param weight_loss: numpy array of shape (C,) where C is the number of classes
    :return:
    """
    weight_loss = np.array(weight_loss)
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    Y_gt = tf.cast(Y_gt, tf.float32)
    # Compute gen dice coef:
    numerator = Y_gt * Y_pred
    numerator = tf.reduce_sum(numerator, axis=(1, 2, 3))
    denominator = Y_gt + Y_pred
    denominator = tf.reduce_sum(denominator, axis=(1, 2, 3))
    gen_dice_coef = tf.reduce_mean(2. * (numerator + smooth_tf) / (denominator + smooth_tf), axis=0)
    loss = -tf.reduce_mean(weight_loss * gen_dice_coef)
    return loss
	

def categorical_focal_loss(Y_pred, Y_gt, gamma, alpha):
    """
     Categorical focal_loss between an output and a target
    :param Y_pred: A tensor of the same shape as `y_pred`
    :param Y_gt: A tensor resulting from a softmax(-1,z,h,w,numclass)
    :param alpha: Sample category weight,which is shape (C,) where C is the number of classes
    :param gamma: Difficult sample weight
    :return:
    """
    weight_loss = np.array(alpha)
    epsilon = 1.e-5
    # Scale predictions so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keepdims=True)
    # Clip the prediction value to prevent NaN's and Inf's
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    # Calculate Cross Entropy
    cross_entropy = -Y_gt * tf.log(output)
    # Calculate Focal Loss
    loss = tf.pow(1 - output, gamma) * cross_entropy
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_mean(weight_loss * loss)
    return loss
	
	
def categorical_dicePcrossentroy(Y_pred, Y_gt, weight, lamda=0.5):
    """
    hybrid loss function from dice loss and crossentroy
    loss=Ldice+lamda*Lfocalloss
    :param Y_pred:A tensor resulting from a softmax(-1,z,h,w,numclass)
    :param Y_gt: A tensor of the same shape as `y_pred`
    :param gamma:Difficult sample weight
    :param alpha:Sample category weight,which is shape (C,) where C is the number of classes
    :param lamda:trade-off between dice loss and focal loss,can set 0.1,0.5,1
    :return:diceplusfocalloss
    """
    weight_loss = np.array(weight)
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    Y_gt = tf.cast(Y_gt, tf.float32)
    # Compute gen dice coef:
    numerator = Y_gt * Y_pred
    numerator = tf.reduce_sum(numerator, axis=(1, 2, 3))
    denominator = Y_gt + Y_pred
    denominator = tf.reduce_sum(denominator, axis=(1, 2, 3))
    gen_dice_coef = tf.reduce_sum(2. * (numerator + smooth_tf) / (denominator + smooth_tf), axis=0)
    loss1 = tf.reduce_mean(weight_loss * gen_dice_coef)
    epsilon = 1.e-5
    # scale preds so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keep_dims=True)
    # manual computation of crossentropy
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    loss = -Y_gt * tf.log(output)
    loss = tf.reduce_mean(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss2 = tf.reduce_mean(weight_loss * loss)
    total_loss = (1 - lamda) * (1 - loss1) + lamda * loss2
    return total_loss
	
	
def categorical_dicePfocalloss(Y_pred, Y_gt, alpha, lamda=0.5, gamma=2.):
    """
    hybrid loss function from dice loss and focalloss
    loss=Ldice+lamda*Lfocalloss
    :param Y_pred:A tensor resulting from a softmax(-1,z,h,w,numclass)
    :param Y_gt: A tensor of the same shape as `y_pred`
    :param gamma:Difficult sample weight
    :param alpha:Sample category weight,which is shape (C,) where C is the number of classes
    :param lamda:trade-off between dice loss and focal loss,can set 0.1,0.5,1
    :return:dicePfocalloss
    """
    weight_loss = np.array(alpha)
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    Y_gt = tf.cast(Y_gt, tf.float32)
    # Compute gen dice coef:
    numerator = Y_gt * Y_pred
    numerator = tf.reduce_sum(numerator, axis=(1, 2, 3))
    denominator = Y_gt + Y_pred
    denominator = tf.reduce_sum(denominator, axis=(1, 2, 3))
    gen_dice_coef = tf.reduce_sum(2. * (numerator + smooth_tf) / (denominator + smooth_tf), axis=0)
    loss1 = tf.reduce_mean(weight_loss * gen_dice_coef)
    epsilon = 1.e-5
    # Scale predictions so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keepdims=True)
    # Clip the prediction value to prevent NaN's and Inf's
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    # Calculate Cross Entropy
    cross_entropy = -Y_gt * tf.log(output)
    # Calculate Focal Loss
    loss = tf.pow(1 - output, gamma) * cross_entropy
    loss = tf.reduce_mean(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss2 = tf.reduce_mean(weight_loss * loss)
    total_loss = (1 - lamda) * (1 - loss1) + lamda * loss2
    return total_loss