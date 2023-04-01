class GraphConvModel(nn.Module):
    def __init__(self, gnn_in, gnn_hidden, gnn_out, gnn_num_layers, mlp_hidden, mlp_num_layers):
        super().__init__()
        # graph neural network
        if gnn_num_layers == 1:
            gnn_layers = [geom_nn.GraphConv(in_channels=gnn_in, out_channels=gnn_out), nn.ReLU()]
        else:
            gnn_layers = [geom_nn.GraphConv(in_channels=gnn_in, out_channels=gnn_hidden), nn.ReLU()]
            for _ in range(gnn_num_layers-2):
                gnn_layers += [geom_nn.GraphConv(in_channels=gnn_hidden, out_channels=gnn_hidden), nn.ReLU()]
            gnn_layers += [geom_nn.GraphConv(in_channels=gnn_hidden, out_channels=gnn_out)]
        self.gnn_layers = nn.ModuleList(gnn_layers)

        # multi-layer perceptron
        if mlp_num_layers == 1:
            mlp_layers = [nn.Linear(gnn_out, 1), nn.ReLU()]
        else:
            mlp_layers = [nn.Linear(gnn_out, mlp_hidden), nn.ReLU()]
            for _ in range(mlp_num_layers-2):
                mlp_layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU()]
            mlp_layers += [nn.Linear(mlp_hidden, 1)]
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
    def forward(self, x, edge_index, batch):
        # gnn message passing
        for layer in self.gnn_layers:
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        
        # gnn graph aggregation and mlp
        x = geom_nn.global_mean_pool(x, batch)
        x = self.mlp_layers(x)
        return x