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


def tversky_loss_3d(Y_gt, Y_pred):
    smooth = 1e-5
    alpha = 0.5
    beta = 0.5
    ones = tf.ones(tf.shape(Y_gt))
    p0 = Y_pred
    p1 = ones - Y_pred
    g0 = Y_gt
    g1 = ones - Y_gt
    num = tf.reduce_sum(p0 * g0, axis=[1, 2, 3])
    den = num + alpha * tf.reduce_sum(p0 * g1, axis=[1, 2, 3]) + \
          beta * tf.reduce_sum(p1 * g0, axis=[1, 2, 3]) + smooth
    tversky = tf.reduce_sum(num / den, axis=1)
    loss = tf.reduce_mean(1 - tversky)
    return loss


def tversky_loss_2d(Y_gt, Y_pred):
    smooth = 1e-5
    alpha = 0.5
    beta = 0.5
    ones = tf.ones(tf.shape(Y_gt))
    p0 = Y_pred
    p1 = ones - Y_pred
    g0 = Y_gt
    g1 = ones - Y_gt
    num = tf.reduce_sum(p0 * g0, axis=[1, 2])
    den = num + alpha * tf.reduce_sum(p0 * g1, axis=[1, 2]) + \
          beta * tf.reduce_sum(p1 * g0, axis=[1, 2]) + smooth
    tversky = tf.reduce_sum(num / den, axis=1)
    loss = tf.reduce_mean(1 - tversky)
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
    multipled = tf.reduce_sum(Y_gt * Y_pred, axis=[1, 2, 3, 4])
    loss = tf.reduce_mean(multipled)
    return loss


def surface_loss_2d(Y_gt, Y_pred):
    multipled = tf.reduce_sum(Y_gt * Y_pred, axis=[1, 2, 3])
    loss = tf.reduce_mean(multipled)
    return loss


x = tf.constant(value=5., shape=(3, 32, 32, 32, 1))
y = tf.constant(value=1., shape=(3, 32, 32, 32, 1))
surface_loss_3d(x, y)
