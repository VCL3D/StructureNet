import OpenEXR
import Imath
import numpy as np
#########      https://github.com/gabrieleilertsen/hdrcnn/issues/1


'''
    Function that saves multidimentional tensor as an EXR image.
    dimensions : CxHxW
'''
def saveTensorEXR(
    tensor          :   np.ndarray,
    filename        :   str
):
    assert len(tensor.shape) == 3, "Tensor should be CxHxW"
    
    c,h,w = tensor.shape
    assert c < 99, "More than 99 channels are not supported"
    assert tensor.dtype == np.float32
    header = OpenEXR.Header(w,h)
    header['channels'] = dict()
    data = dict()

    for i in range(c):
        _id = ("0" if (i<10) else "") + str(i)
        
        header['channels'][_id] = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
        data[_id] = tensor[i].tobytes()
    
    
    file = OpenEXR.OutputFile(filename, header)
    file.writePixels(data)
    file.close()


