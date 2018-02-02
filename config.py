class Config(object):
    init_scale = 0.04
    learning_rate =  0.1
    dropout_keep_prob = 0.75
    max_grad_norm = 10
    num_layers = 2
    sequence_length = 300
    hidden_size = 256
    embed_size = 512
    max_epoch = 6
    keep_prob = 0.75
    decay_rate = 0.5
    batch_size = 64
    use_embedding = True
    is_training = True
    num_classes = 14
    vocab_size = 100000

class TestConfig(object):
    init_scale = 0.04
    learning_rate =  0.1
    dropout_keep_prob = 0.75
    max_grad_norm = 10
    num_layers = 2
    sequence_length = 300
    hidden_size = 256
    embed_size = 512
    max_epoch = 1
    keep_prob = 0.75
    decay_rate = 0.5
    batch_size = 64
    use_embedding = True
    is_training = False
    num_classes = 14
    vocab_size = 100000
