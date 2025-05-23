import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import LSTM

class GCN_LSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edge_features, num_layers, dropout):
        super(GCN_LSTM, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, dropout=dropout)
        self.conv2 = GCNConv(hidden_channels, out_channels, dropout=dropout)
        #self.conv3 = GCNConv(hidden_channels, out_channels, dropout=dropout)
        #self.conv4 = GCNConv(hidden_channels, out_channels, dropout=dropout)
        self.lstm = LSTM(out_channels, hidden_channels, num_layers, batch_first=True, dropout=dropout)
        self.edge_linear = torch.nn.Linear(num_edge_features, hidden_channels)
        self.classifier = torch.nn.Linear(3*hidden_channels, 1)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_features, hidden=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)
        #x = self.conv4(x, edge_index)
        
        x = x.unsqueeze(0)  # Add batch dimension for LSTM
        if hidden is None:
            hidden = (torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size),
                      torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size))
            lstm_out, hidden = self.lstm(x, hidden)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        lstm_out = lstm_out.squeeze(0)  # Remove batch dimension
        
        edge_embeddings = self.edge_linear(edge_features)
        src = lstm_out[edge_index[0]]
        dst = lstm_out[edge_index[1]]
        #combined_embeddings = src + dst + edge_embeddings
        combined_embeddings = torch.cat([src,dst,edge_embeddings], dim=-1)
        edge_pred = self.classifier(combined_embeddings)
        return edge_pred, hidden, src, dst
  
