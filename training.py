import torch
import utils
from tqdm.autonotebook import tqdm
import numpy as np
import os
from loss_functions import compute_psnr
import torch.nn as nn

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, summary_fn):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[200, 400, 600, 800], gamma=0.5)

    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    total_steps = 0
    best_psnr = 0
    best_loss = 10

    n_iterations = epochs * (len(train_dataloader) / train_dataloader.batch_size)

    with tqdm(total=n_iterations) as pbar:
        train_losses = []

        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save({"model": model.state_dict()},
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))
            
            for step, (input, coords, gt) in enumerate(train_dataloader):
            
                model_output = model(input, coords)

                # print("Model input min max", model_input.min(), model_input.max())
                # print("GT coords min max", gt_coords.min(), gt_coords.max())
                # print("GT values min max", gt_values.min(), gt_values.max())

                loss = loss_fn(model_output['model_out'], gt)
                train_loss.add(loss.item())

                psnr = compute_psnr(model_output['model_out'], gt)

                if loss.item() < best_loss:
                    torch.save({'epoch': total_steps,
                                        'model': model.state_dict(),
                                        'optimizer': optim.state_dict(),
                                        'scheduler': scheduler.state_dict(),
                                        # 'latent_grid': model.latent_grid,    
                                        }, os.path.join(checkpoints_dir, 'model_best.pth'))

                    best_loss = loss.item()
                
                # print("Loss", loss.item())

                optim.zero_grad()
                loss.backward()
                optim.step()

                if not total_steps % steps_til_summary:
                     # total_loss = round(sum(train_loss) / len(train_loss), 5)
                     tqdm.write("Epoch {}, Total loss {}, PSNR {}".format(epoch, train_loss.item(), psnr))

                pbar.update(1)
                total_steps += 1

            scheduler.step()

        torch.save({'epoch': total_steps,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),  
                    # 'latent_grid': model.latent_grid,  
                    }, os.path.join(checkpoints_dir, f'model_final.pth'))
        
        return psnr

