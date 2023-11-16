def image_mse(mask, model_output, gt):
    if mask is None:


        # print("Min max model_output", model_output.min(), model_output.max())
        # print("Min max gt", gt.min(), gt.max())

        return {'img_loss': ((model_output- gt) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output - gt) ** 2).mean()}

