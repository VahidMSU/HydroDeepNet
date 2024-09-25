def get_model_by_name(name, in_channels_dict, out_channels, aggregation, edge_attr_dict, **kwargs):
    if name == 'SharedLinearPReLUModel':
        from libs.SharedLinearPReLUModel import SharedLinearPReLUModel
        return SharedLinearPReLUModel(in_channels_dict, out_channels, aggregation, edge_attr_dict, **kwargs)
    
    elif name == 'DeepGraphSAGEModel':
        from libs.DeepGraphSAGEModel import DeepGraphSAGEModel
        return DeepGraphSAGEModel(in_channels_dict, out_channels, aggregation, edge_attr_dict, **kwargs)
    
    elif name == 'SeparateLinearReLUModel':
        from libs.SeparateLinearReLUModel import SeparateLinearReLUModel
        return SeparateLinearReLUModel(in_channels_dict, out_channels, aggregation, edge_attr_dict, **kwargs)
    elif name == 'SeparateLinearModel':
        from libs.SeparateLinearModel import SeparateLinearModel
        return SeparateLinearModel(in_channels_dict, out_channels, aggregation, edge_attr_dict, **kwargs)
    
    elif name == 'DeepGatedEdgePReLUGNN':
        from libs.DeepGatedEdgePReLUGNN import DeepGatedEdgePReLUGNN
        return DeepGatedEdgePReLUGNN(in_channels_dict, out_channels, aggregation, edge_attr_dict, **kwargs)

    elif name == 'GatedEdgePReLUGNN':
        from libs.GatedEdgePReLUGNN import GatedEdgePReLUGNN
        return GatedEdgePReLUGNN(in_channels_dict, out_channels, aggregation, edge_attr_dict, **kwargs)
    
    elif name == 'GatedEdgeEmbeddingPReLUGNN':
        from libs.GatedEdgeEmbeddingPReLUGNN import GatedEdgeEmbeddingPReLUGNN
        return GatedEdgeEmbeddingPReLUGNN(in_channels_dict, out_channels, aggregation, edge_attr_dict, **kwargs)
    
    elif name == 'AttentionEdgePReLUGNN':
        from libs.AttentionEdgePReLUGNN import AttentionEdgePReLUGNN
        return AttentionEdgePReLUGNN(in_channels_dict, out_channels, aggregation, edge_attr_dict, **kwargs)
    
    else:
        raise ValueError(f"Unknown model name: {name}")

