import torch
import torch.nn as nn
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
import gc

from src.models.baseModel import BaseModel

from src.core.generalUtils import weightLocation, nameModelRun


## https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books

## Review
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class _DeepLOB_PT(nn.Module):
    """
    Description:
        This is the original deepLOB model build with PyRotch, we are using it as the base mode for DeepLOB_PT
        https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
        -> DO NOT USE THIS MODEL DIRECTLY, USE THE WRAPPER CLASS DeepLOB_PT DEFINED BELOW
    """
    def __init__(self):
        super().__init__()
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        
        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, 3)
        
    def forward(self, x) -> torch.tensor:
        # Accept input as (batch, height, width, channels) or (height, width, channels)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # If shape is (batch, 40, 100, 1), transpose to (batch, 100, 40, 1)
        # REVIEW!
        if x.shape[1] == 40 and x.shape[2] == 100:
            x = x.permute(0, 2, 1, 3)
        if x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError("Input must have shape (batch, height, width, 1) or (height, width, 1)")

        h0 = torch.zeros(1, x.shape[0], 64).to(device)
        c0 = torch.zeros(1, x.shape[0], 64).to(device)
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)  
        
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)
        
        return forecast_y
        
class DeepLOB_PT(BaseModel):
    """
    Description:
        Wrapper class around _DeepLOB_PT. Use this model.
    """
    name = 'deepLOB_PT'
    def __init__(self):
        super().__init__()
        self.name = DeepLOB_PT.name  
        self.weightsFileFormat = "pth"
        self.model = _DeepLOB_PT()
        
    def train(self, x : tensor, y : tensor, batchSize : int, numEpoch : int, learningRate : float = 1e-3):
        self.model.to(device)
        print("Moved model to device:", device)
        self.model.train()
        x, y = x.to(device), y.to(device)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(numEpoch):
            total_loss = 0
            correct = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model.forward(xb)
                correct += (preds.argmax(dim=1) == yb.argmax(dim=1)).sum().item()
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(dataset)
            avg_correct = correct / len(dataset)
            print(f"Epoch {epoch+1}/{numEpoch}, Loss: {avg_loss:.4f}, Correct: {avg_correct:.4f}")
        
        del x, y  # remove reference to tensor on GPU
        gc.collect()  # force Python garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release unused GPU memory back to the OS
            

    def predict(self, x, y = None, batch_size: int = 128) -> torch.tensor:
        """
        Runs a forward pass in evaluation mode (no gradients, no dropout), processing in batches to avoid OOM.
        Args:
            x (torch.Tensor): Input tensor of shape [batch, height, width, channels]
            batch_size (int): Batch size for prediction
        Returns:
            torch.Tensor: Model predictions (probabilities)
        """
        self.model.eval()
        preds_list = []
        x = x.to(device)
        if y is not None:
            y = y.to(device)
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                xb = x[i:i+batch_size]
                preds = self.model.forward(xb)
                preds_list.append(preds.cpu())
        # del x  # remove reference to tensor on GPU
        # gc.collect()  # force Python garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release unused GPU memory back to the OS

        preds = torch.cat(preds_list, dim=0)
        return preds
    
    def saveWeights(self, run_id: str = "") -> None:
        fileName = nameModelRun(run_id)
        torch.save(self.model.state_dict(), weightLocation(self, runName=fileName))
        
    def loadFromWeights(self, weightsPath : str) -> None:
        self.model.load_state_dict(torch.load(weightsPath, map_location=device))
        self.model.to(device)
        
if __name__ == "__main__":
    model = DeepLOB_PT()
    model.saveWeights()