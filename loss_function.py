import tensorflow as tf


def dice_loss_3d(Y_gt, Y_pred):
    Z, H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    pred_flat = tf.reshape(Y_pred, [-1, H * W * C * Z])
    true_flat = tf.reshape(Y_gt, [-1, H * W * C * Z])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
    loss = -tf.reduce_mean(intersection / denominator)
    return loss


def dice_loss_2d(Y_gt, Y_pred):
    H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    pred_flat = tf.reshape(Y_pred, [-1, H * W * C])
    true_flat = tf.reshape(Y_gt, [-1, H * W * C])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
    loss = -tf.reduce_mean(intersection / denominator)
    return loss


def twersky_loss_3d(Y_gt, Y_pred):
    smooth = 1e-5
    alpha = 0.5
    beta = 0.5
    ones = tf.ones(tf.shape(Y_gt))
    p0 = Y_pred
    p1 = ones - Y_pred
    g0 = Y_gt
    g1 = ones - Y_gt
    num = tf.reduce_sum(p0 * g0, axis=(0, 1, 2, 3))
    den = num + alpha * tf.reduce_sum(p0 * g1, axis=(0, 1, 2, 3)) + \
          beta * tf.reduce_sum(p1 * g0, axis=(0, 1, 2, 3)) + smooth
    T = tf.reduce_sum(num / den)
    Ncl = tf.cast(tf.shape(Y_gt)[-1], 'float32')
    loss = Ncl - T
    return loss


def twersky_loss_2d(Y_gt, Y_pred):
    smooth = 1e-5
    alpha = 0.5
    beta = 0.5
    ones = tf.ones(tf.shape(Y_gt))
    p0 = Y_pred
    p1 = ones - Y_pred
    g0 = Y_gt
    g1 = ones - Y_gt
    num = tf.reduce_sum(p0 * g0, axis=(0, 1, 2))
    den = num + alpha * tf.reduce_sum(p0 * g1, axis=(0, 1, 2)) + \
          beta * tf.reduce_sum(p1 * g0, axis=(0, 1, 2)) + smooth
    T = tf.reduce_sum(num / den)
    Ncl = tf.cast(tf.shape(Y_gt)[-1], 'float32')
    loss = Ncl - T
    return loss


def generalized_dice_loss_3d(Y_gt, Y_pred):
    smooth = 1e-5
    Ncl = tf.shape(Y_pred)[-1]
    w = tf.zeros(shape=(Ncl,))
    w = tf.reduce_sum(Y_gt, axis=(0, 1, 2, 3))
    w = 1 / (w ** 2 + smooth)

    numerator = Y_gt * Y_pred
    numerator = w * tf.reduce_sum(numerator, axis=(0, 1, 2, 3, 4))
    numerator = tf.reduce_sum(numerator)

    denominator = Y_pred + Y_gt
    denominator = w * tf.reduce_sum(denominator, axis=(0, 1, 2, 3, 4))
    denominator = tf.reduce_sum(denominator)

    gen_dice_coef = 2 * numerator / (denominator + smooth)
    loss = 1 - gen_dice_coef
    return loss


def generalized_dice_loss_2d(Y_gt, Y_pred):
    smooth = 1e-5
    Ncl = tf.shape(Y_pred)[-1]
    w = tf.zeros(shape=(Ncl,))
    w = tf.reduce_sum(Y_gt, axis=(0, 1, 2))
    w = 1 / (w ** 2 + smooth)

    numerator = Y_gt * Y_pred
    numerator = w * tf.reduce_sum(numerator, axis=(0, 1, 2, 3))
    numerator = tf.reduce_sum(numerator)

    denominator = Y_pred + Y_gt
    denominator = w * tf.reduce_sum(denominator, axis=(0, 1, 2, 3))
    denominator = tf.reduce_sum(denominator)

    gen_dice_coef = 2 * numerator / (denominator + smooth)
    loss = 1 - gen_dice_coef
    return loss
