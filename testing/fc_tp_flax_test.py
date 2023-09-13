from models.fc_tp_flax import FullyConnectedTensorProduct
import e3nn_jax as e3nn
import jax

def test_fully_connected_tensor_product(keys):

    model = FullyConnectedTensorProduct("10x0e + 1e")
    x1 = e3nn.normal("5x0e + 1e", keys, (10,))
    x2 = e3nn.normal("3x1e + 2x0e", keys, (20, 1))

    params = model.init(keys, x1, x2)
    x3 = model.apply(params, x1, x2)
    print(x3.irreps == e3nn.Irreps("10x0e + 1e"))
    assert x3.irreps == e3nn.Irreps("10x0e + 1e")
    assert x3.shape[:-1] == (20, 10)

test_fully_connected_tensor_product(jax.random.PRNGKey(6))