def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output- gt) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output - gt) ** 2).mean()}

