import traceback
from convert.torch2caffe.caffe_net import Net

def pair_process(item, strict_one=False):
    if hasattr(item, '__iter__'):
        for i in item:
            if i != item[0]:
                if strict_one:
                    raise ValueError(
                        "number in item {} must be the same".format(item))
                else:
                    return item
        return item
    return [item, item]

def pair_reduce(item):
    if hasattr(item, '__iter__'):
        for i in item:
            if i != item[0]:
                return item
        return [item[0]]
    return [item]

class Blob_LOG():
    def __init__(self):
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

# 转换原理解析：通过记录
class TransLog(object):
    def __init__(self):
        """
        doing init() with inputs Variable before using it
        """
        self.layers = {}
        self.detail_layers = {}
        self.detail_blobs = {}
        self._blobs = Blob_LOG()
        self._blobs_data = []
        self.debug = True

        self.NET_INITTED = False
        self.layer_names = {}

        self.pytorch_layer_name = None
        self.simple_name = True

        self.ckpt = 'yolodet.pth'
        self.output_blob_pre = ''
        self.cnet = None

        self.merge_bn = False
        self.layer_type_order = []
        self.layer_name_order = []

    # def init(self, inputs):
    #     """
    #     :param inputs: is a list of input variables
    #     """

    #     self.cnet = Net(ckpt=self.ckpt, output_blob_pre=self.output_blob_pre)

    #     self.add_blobs(inputs, name='data', with_num=False)
    def init(self):
        """
        :param inputs: is a list of input variables
        """

        self.cnet = Net(ckpt=self.ckpt, output_blob_pre=self.output_blob_pre)

    def add_layer(self, name='layer'):

        self.layer_type_order.append(name)

        if not self.simple_name:
            if self.pytorch_layer_name:
                name = self.pytorch_layer_name

        if name in self.layers:
            self.layer_name_order.append(name)
            return self.layers[name]

        if name not in self.detail_layers.keys():
            self.detail_layers[name] = 0

        self.detail_layers[name] += 1
        name = '{}{}'.format(name, self.detail_layers[name])
        self.layers[name] = name

        if self.debug:
            print("\t{} was added to layers".format(self.layers[name]))

        self.layer_name_order.append(name)
        return self.layers[name]

    def add_blobs(self, blobs, name='blob', with_num=True):
        rst = []
        for blob_idx, blob in enumerate(blobs):
            self._blobs_data.append(
                blob)  # to block the memory address be rewrited
            blob_id = int(id(blob))
            if name not in self.detail_blobs.keys():
                self.detail_blobs[name] = 0
            self.detail_blobs[name] += 1
            if with_num and self.detail_blobs[name]!=1:
                rst.append('{}{}'.format(name, self.detail_blobs[name]))
            else:
                rst.append('{}'.format(name))
            if self.debug:
                print("\ttop blob {}: {}:{}, dim:{}, was added to blobs".format(blob_idx+1, blob_id, rst[-1], [int(dim) for dim in blob.shape]))
            # print('Add blob {} : {}'.format(rst[-1].center(21),blob.size()))
            self._blobs[blob_id] = rst[-1]
        return rst

    def blobs(self, var):
        var_id = id(var)
        # if self.debug:
        #     print("{}:{} getting".format(var, self._blobs[var]))
        try:
            return self._blobs[var_id]
        except:
            raise Exception("CANNOT FOUND blob {}, blob shape: {}".format(var_id, var.shape))
            # return None

trans_log = TransLog()


# 核心组件，通过该类，实现对torch的function中的operators的输入，输出以及参数的读取
class Rp(object):
    def __init__(self, raw, replace, **kwargs):
        # replace the raw function to replace function
        self.obj = replace
        self.raw = raw

    def __call__(self, *args, **kwargs):
        if not trans_log.NET_INITTED:
            return self.raw(*args, **kwargs)
        
        input_layer = True
        for stack in traceback.walk_stack(None):
            if 'self' in stack[0].f_locals:
                layer = stack[0].f_locals['self']
                if layer in trans_log.layer_names:
                    input_layer = False
                    trans_log.pytorch_layer_name = trans_log.layer_names[layer]
                    if trans_log.layer_names[layer]:
                        print(trans_log.layer_names[layer])
                    else:
                        print('input')
                    break
        
        if input_layer:
            print('input')

        out = self.obj(self.raw, *args, **kwargs)
        # if isinstance(out,Variable):
        #     out=[out]
        return out


if __name__ == "__main__":

    def nothing():
        pass
    
    import torch
    import torchvision.ops as ops
    trans_log.NET_INITTED = True
    
    torch.ops.torchvision.roi_pool = Rp(torch.ops.torchvision.roi_pool, nothing)

    pool = ops.RoIPool(spatial_scale=0.5, output_size=7)

    input = torch.ones([1,3,256,256])
    box = torch.Tensor([[0,0,0,32,32]])
    pool(input, box)
