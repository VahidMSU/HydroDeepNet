def select_model(config, number_of_channels, device):
    
    if config['model'] == 'SimpleCNN':
        from GeoCNN.SimpleCNN import SimpleCNN
        model = SimpleCNN(num_channel=number_of_channels, num_classes=1).to(device)
    elif config['model'] == 'AdvancedCNN':
        from GeoCNN.AdvancedCNN import AdvancedCNN
        model = AdvancedCNN(num_channels=number_of_channels, num_classes=1).to(device)
    elif config['model'] == 'ModifiedResNet':
        from GeoCNN.ResNet import ModifiedResNet
        model = ModifiedResNet(num_input_channels=number_of_channels, output_size=(config['desired_rows'], config['desired_cols'])).to(device) 
    elif config['model'] == 'ModifiedResNetUNet':
        from GeoCNN.ResNetUnet import ModifiedResNetUNet
        model = ModifiedResNetUNet(num_input_channels=number_of_channels).to(device)
    elif config['model'] == 'AdvancedRegressorCNN':
        from GeoCNN.AdvancedRegressionCNN import AdvancedRegressorCNN
        model = AdvancedRegressorCNN(num_channels=number_of_channels).to(device)
    elif config['model'] == 'CNNTransformerRegressor':
        from GeoCNN.CNNTransformerRegressor import CNNTransformerRegressor
        model = CNNTransformerRegressor(
            num_channels=number_of_channels,
            embed_size=config['embed_size'],
            num_heads=config['num_heads'],
            forward_expansion=config['forward_expansion'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_classes=1
        ).to(device)
    elif config['model'] == 'TransformerCNN':
        from GeoCNN.TransformerCNN import TransformerCNN
        model = TransformerCNN(
            number_of_channels,  # Input channels based on the shape of the data
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            forward_expansion=config['forward_expansion'],
            verbose=config['verbose']
        ).to(device)

    return model