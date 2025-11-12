import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torchinfo import summary
from src.utils.logger import logger

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x

if __name__ == "__main__":
    model_logger = logger.getChild(__name__)
    try:
        input_example_nodes = torch.rand((10, 10))
        input_example_edges = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        ], dtype=torch.long)
        
        model = GCN(10, 2)
        
        logits = model(input_example_nodes, input_example_edges)  

        model_logger.info(f"Logist shape: {logits.shape}") 
        model_logger.info("Model architecture:\n%s", model)
    except Exception as e:
        model_logger.error(f"Error during model creation: {e}")
