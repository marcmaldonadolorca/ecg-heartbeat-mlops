import torch

from ecg_mlops.data import N_TIMESTEPS
from ecg_mlops.model import ECGCNN


def test_cnn_forward_shape():
    model = ECGCNN(n_classes=5, base_filters=8, dropout=0.1)
    x = torch.randn(4, 1, N_TIMESTEPS)

    logits = model(x)

    assert logits.shape == (4, 5)


def test_cnn_backward_has_gradients():
    model = ECGCNN(n_classes=5, base_filters=8, dropout=0.1)
    x = torch.randn(4, 1, N_TIMESTEPS)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    loss = torch.nn.CrossEntropyLoss()(model(x), y)
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert all(grad is not None for grad in grads)

