config = {
    'log_path': '/vtt/C/log',
    'tokenizer': 'nltk',
    'batch_sizes': (12, 12, 12),
    'lower': True,
    'cache_image_vectors': True,
    'split_ratio': [8, 1, 1],
    'image_path': '/vtt/C/data/images',
    'data_path': '/vtt/C/data/FriendsQA.jsonl',
    'vocab_pretrained': "glove.6B.300d",
    'shots_only': True,
    'feature_pooling_method': 'mean',
    'max_epochs': 20,
    'allow_empty_images': False,
    'num_workers': 40,
    'model_name': 'acc_model',
    'image_dim': 512,  # hardcoded for ResNet50
    'n_dim': 256,
    'layers': 3,
    'dropout': 0.5,
    'learning_rate': 0.001,
    'loss_name': 'cross_entropy_loss',
    'optimizer': 'adagrad',
    # 'metrics': ['bleu', 'rouge'],
    'metrics': [],
    'log_cmd': False,
    'ckpt_path': '/vtt/C/data/ckpt',
    'ckpt_name': None,
    'device': 'cuda'
}


debug_options = {
    # 'image_path': './data/images/samples',
}

log_keys = [
    'model_name',
    'feature_pooling_method',
]
