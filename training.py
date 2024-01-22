import torch
import utils
from tqdm.autonotebook import tqdm
import numpy as np
import os
from loss_functions import compute_psnr
import torch.nn as nn
from ignite.metrics import PSNR


import wandb

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, summary_fn, opt):

    run = wandb.init(project="continuous-volumes", group=opt.experiment_name, config=opt)

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2000, 4000, 6000, 8000], gamma=0.5)

    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    total_steps = 0

    train_losses = []

    psnr_metric = PSNR(data_range=1.0)  # Use data_range=255 for images in [0, 255]


    for epoch in tqdm(range(epochs)):
        if not epoch % epochs_til_checkpoint and epoch:
            torch.save({"model": model.state_dict()},
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))
            
        for step, (input, coords, gt, filename) in enumerate(train_dataloader):
            
            model_output = model(input, coords)

            loss = loss_fn(model_output['model_out'], gt)
            train_loss.add(loss.item())

            psnr_metric.update((model_output['model_out'], gt))
            psnr = psnr_metric.compute()
            psnr_metric.reset()

            # psnr = compute_psnr(model_output['model_out'], gt)

            wandb.log({"train_loss": loss.item(), "train_psnr": psnr})
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            if not total_steps % steps_til_summary:
                tqdm.write("Epoch {}, Total loss {}, PSNR {}".format(epoch, train_loss.item(), psnr))

                torch.save({'epoch': total_steps,
                                    'model': model.state_dict(),
                                    'optimizer': optim.state_dict(),
                                    'scheduler': scheduler.state_dict(),
                                    }, os.path.join(checkpoints_dir, 'model_latest.pth'))

            total_steps += 1
        scheduler.step()

    wandb.finish()

    torch.save({'epoch': total_steps,
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),  
                }, os.path.join(checkpoints_dir, f'model_final.pth'))
        
    return psnr

