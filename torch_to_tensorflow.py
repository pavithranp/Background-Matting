#from torch.autograd import Variable
#from torch import nn
#import torch
#from networks import ResnetConditionHR
from collections import OrderedDict
import onnx
from onnx_tf.backend import prepare


def copy_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

if __name__== "__main__":
    """    # Load the trained model from file
        netM = ResnetConditionHR(input_nc=(3, 3, 1, 4), output_nc=4, n_blocks1=7, n_blocks2=3)
        # netM=nn.DataParallel(netM)
        netM.load_state_dict(copy_dict(torch.load('Models/real-hand-held/netG_epoch_12.pth')))
        dummy_input1 = Variable(torch.randn(1, 1, 512, 512))
        dummy_input2 = Variable(
            torch.randn(1, 3, 512, 512))  # one black and white 28 x 28 picture will be the input to the model
        dummy_input3 = Variable(torch.randn(1, 4, 512, 512))
    
        torch.onnx.export(netM, (dummy_input2, dummy_input2, dummy_input1, dummy_input3), "matte.onnx")
    """



    # Load the ONNX file
    model = onnx.load('Models/matte.onnx')

    # Import the ONNX model to Tensorflow
    tf_rep = prepare(model)
    # Input nodes to the model
    print('inputs:', tf_rep.inputs)

    # Output nodes from the model
    print('outputs:', tf_rep.outputs)

    # All nodes in the model
    print('tensor_dict:')
    print(tf_rep.tensor_dict)
