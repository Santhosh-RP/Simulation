import math
from pathlib import Path

import torch
import torch.nn as nn

# from here import RSSM
class RSSM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, num_layers=1):
        super(RSSM, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.h = None

    def forward(self, x, online=False):
        if online is True:
            x, self.h = self.rnn(x, self.h)
        else:
            x, self.h = self.rnn(x)
        x = self.fc(x)
        return x

BATCH_SIZE=54
EPOCHS = 10000
LEARNING_RATE = 0.001

PUBLISH = True

if PUBLISH:
    import wandb

HIDDEN_SIZE = 2048
NUM_LAYERS = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    rssm = RSSM(164+2, HIDDEN_SIZE, 164, num_layers=NUM_LAYERS).to(device)

    train = torch.load('rssm_dataset_train.pth')
    val = torch.load('rssm_dataset.pth')

    optimizer = torch.optim.Adam(rssm.parameters(), lr=LEARNING_RATE)
    loss = nn.MSELoss()

    task_id = f"rssm_{HIDDEN_SIZE}h_{NUM_LAYERS}l"
    if PUBLISH:
        wandb.init(project="RSSM", notes=f"r{task_id}", name=f"r{task_id}")
        wandb.config = {
            "task_id": task_id,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": EPOCHS,
        }

    best_val_value = None
    best = None
    for epoch in range(EPOCHS):
        if Path("stop").is_file():
            print('Stopping due to "stop" file')
            break
        # Train
        rssm.train()
        perm = torch.randperm(train.shape[0])
        avg_loss = 0
        grads = {}
        for batch in range(int(math.floor(train.shape[0]/BATCH_SIZE))):
            idx = perm[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            batch_data = train[idx]
            X = batch_data[:, 0:-1, :].to(device)
            TY = batch_data[:, 1:,   :-2].to(device) # -2 because the action is of size 2 (two wheels)
            y = rssm(X)
            L = loss(y, TY)
            optimizer.zero_grad()
            L.backward()
            for name, param in rssm.named_parameters():
                if not name in grads.keys():
                    grads[name+"_grad"] = 0
                grads[name+"_grad"] += torch.sum(param.abs())
            avg_loss += L
            optimizer.step()
        avg_loss /= (batch+1)/BATCH_SIZE
        # Validate
        rssm.eval()
        with torch.no_grad():
            val_avg_loss = 0
            for batch in range(int(math.floor(val.shape[0]/BATCH_SIZE))):
                batch_data = val[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                X = batch_data[:, 0:-1, :].to(device)
                TY = batch_data[:, 1:,   :-2].to(device) # -2 because the action is of size 2 (two wheels)
                y = rssm(X)
                L = loss(y, TY)
                val_avg_loss += L
            val_avg_loss /= (batch+1)/BATCH_SIZE
   
        print(f"\rRSSM Epoch: {epoch}/{EPOCHS} \tTL={avg_loss} \tVL={val_avg_loss}")

        epoch += 1
        save = False
        if best_val_value is None or val_avg_loss < best_val_value:
            best_val_value = val_avg_loss
            best = rssm.state_dict()
            if epoch > 5:
                torch.save(rssm.state_dict(), f'train_rssm_best_{str(epoch).zfill(6)}.pth')
            save = True
        if PUBLISH:
            wandb.log(dict({"epoch": epoch, "train_loss": avg_loss, "val_loss": val_avg_loss, 'save': save, "best": best, **grads}))

   
    if PUBLISH:
        wandb.finish()


