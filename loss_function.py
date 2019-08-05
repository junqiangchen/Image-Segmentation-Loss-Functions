import tensorflow as tf


def dice_loss_3d(Y_gt, Y_pred):
    Z, H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    pred_flat = tf.reshape(Y_pred, [-1, H * W * C * Z])
    true_flat = tf.reshape(Y_gt, [-1, H * W * C * Z])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
    loss = 1 - tf.reduce_mean(intersection / denominator)
    return loss


def dice_loss_2d(Y_gt, Y_pred):
    H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    pred_flat = tf.reshape(Y_pred, [-1, H * W * C])
    true_flat = tf.reshape(Y_gt, [-1, H * W * C])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
    loss = 1 - tf.reduce_mean(intersection / denominator)
    return loss


def tversky_loss_3d(Y_gt, Y_pred, alpha=0.7):
    Z, H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    y_pred_pos = tf.reshape(Y_pred, [-1, H * W * C * Z])
    y_true_pos = tf.reshape(Y_gt, [-1, H * W * C * Z])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos, axis=1)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos), axis=1)
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos, axis=1)
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    loss = 1 - tf.reduce_mean(tversky)
    return loss


def tversky_loss_2d(Y_gt, Y_pred, alpha=0.7):
    H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    y_pred_pos = tf.reshape(Y_pred, [-1, H * W * C])
    y_true_pos = tf.reshape(Y_gt, [-1, H * W * C])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos, axis=1)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos), axis=1)
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos, axis=1)
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    loss = 1 - tf.reduce_mean(tversky)
    return loss


def focal_tversky_3d(Y_gt, Y_pred, alpha=0.7, gamma=0.75):
    Z, H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    y_pred_pos = tf.reshape(Y_pred, [-1, H * W * C * Z])
    y_true_pos = tf.reshape(Y_gt, [-1, H * W * C * Z])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos, axis=1)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos), axis=1)
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos, axis=1)
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    loss = 1 - tf.reduce_mean(tversky)
    loss = tf.pow(loss, gamma)
    return loss


def focal_tversky_2d(Y_gt, Y_pred, alpha=0.7, gamma=0.75):
    H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    y_pred_pos = tf.reshape(Y_pred, [-1, H * W * C])
    y_true_pos = tf.reshape(Y_gt, [-1, H * W * C])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos, axis=1)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos), axis=1)
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos, axis=1)
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    loss = 1 - tf.reduce_mean(tversky)
    loss = tf.pow(loss, gamma)
    return loss


def generalised_dice_loss_3d(Y_gt, Y_pred):
    smooth = 1e-5
    w = tf.reduce_sum(Y_gt, axis=[1, 2, 3])
    w = 1 / (w ** 2 + smooth)

    numerator = Y_gt * Y_pred
    numerator = w * tf.reduce_sum(numerator, axis=[1, 2, 3])
    numerator = tf.reduce_sum(numerator, axis=1)

    denominator = Y_pred + Y_gt
    denominator = w * tf.reduce_sum(denominator, axis=[1, 2, 3])
    denominator = tf.reduce_sum(denominator, axis=1)

    gen_dice_coef = 2 * numerator / (denominator + smooth)
    loss = tf.reduce_mean(1 - gen_dice_coef)
    return loss


def generalised_dice_loss_2d_ein(Y_gt, Y_pred):
    Y_gt = tf.cast(Y_gt, 'float32')
    Y_pred = tf.cast(Y_pred, 'float32')
    w = tf.einsum("bwhc->bc", Y_gt)
    w = 1 / ((w + 1e-10) ** 2)
    intersection = w * tf.einsum("bwhc,bwhc->bc", Y_pred, Y_gt)
    union = w * (tf.einsum("bwhc->bc", Y_pred) + tf.einsum("bwhc->bc", Y_gt))

    divided = 1 - 2 * (tf.einsum("bc->b", intersection) + 1e-10) / (tf.einsum("bc->b", union) + 1e-10)

    loss = tf.reduce_mean(divided)
    return loss


def generalised_dice_loss_2d(Y_gt, Y_pred):
    smooth = 1e-5
    w = tf.reduce_sum(Y_gt, axis=[1, 2])
    w = 1 / (w ** 2 + smooth)

    numerator = Y_gt * Y_pred
    numerator = w * tf.reduce_sum(numerator, axis=[1, 2])
    numerator = tf.reduce_sum(numerator, axis=1)

    denominator = Y_pred + Y_gt
    denominator = w * tf.reduce_sum(denominator, axis=[1, 2])
    denominator = tf.reduce_sum(denominator, axis=1)

    gen_dice_coef = 2 * numerator / (denominator + smooth)
    loss = tf.reduce_mean(1 - gen_dice_coef)
    return loss


def surface_loss_3d(Y_gt, Y_pred):
    multipled = tf.reduce_sum(Y_gt * Y_pred, axis=[0,1, 2, 3, 4])
    loss = tf.reduce_mean(multipled)
    return loss


def surface_loss_2d(Y_gt, Y_pred):
    multipled = tf.reduce_sum(Y_gt * Y_pred, axis=[0,1, 2, 3])
    loss = tf.reduce_mean(multipled)
    return loss


def focal_loss_sigmodv1(Y_gt, Y_pred,alpha=0.25, gamma=2):
   epsilon = 1e-5
   pt_1 = tf.where(tf.equal(Y_gt, 1), Y_pred, tf.ones_like(Y_pred))
   pt_0 = tf.where(tf.equal(Y_gt, 0), Y_pred, tf.zeros_like(Y_pred))
   # clip to prevent NaN's and Inf's
   pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
   pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)
   loss_1 = -alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)
   loss_0 = -(1 - alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0)
   loss = tf.reduce_sum(loss_1 + loss_0)
   return loss


def focal_loss_sigmodv2(Y_gt, Y_pred,alpha=0.25, gamma=2):
   epsilon = 1e-5
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    logits = tf.log(y_pred / (1 - y_pred))
    weight_a = alpha * tf.pow((1 - y_pred), gamma) * y_true
    weight_b = (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
    loss = tf.log1p(tf.exp(-logits)) * (weight_a + weight_b) + logits * weight_b
    return tf.reduce_sum(loss)
   return loss



x = tf.constant(value=5., shape=(3, 32, 32, 32, 1))
y = tf.constant(value=1., shape=(3, 32, 32, 32, 1))
surface_loss_3d(x, y)
