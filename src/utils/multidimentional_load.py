import OpenEXR
import Imath
import numpy as np
#########      https://github.com/gabrieleilertsen/hdrcnn/issues/1

def loadTensorExr(
    filename        :   str
):
    file = OpenEXR.InputFile(filename)
    header = file.header()
    w,h,c = header["dataWindow"].max.x + 1, header["dataWindow"].max.y + 1, len(header["channels"])

    list = []
    for channel in header["channels"]:
        data = file.channel(channel)
        x = np.transpose(np.frombuffer(data, dtype = np.float32).reshape(1,h,w), (2,1,0))
        list.append(x)

    # import cv2
    # normals_view = np.transpose(255 * (normals - normals.min())/(normals.max() - normals.min()), (1,2,0))
    # cv2.imshow("d", normals_view.astype(np.uint8))
    # cv2.waitKey(0)

    file.close()
    return list

    # header = OpenEXR.Header(w,h)
    # header['channels'] = dict()
    # data = dict()

    # for i in range(c):
    #     _id = ("0" if (i<10) else "") + str(i)
        
    #     header['channels'][_id] = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
    #     data[_id] = tensor[i].tobytes()
    
    
    # file = OpenEXR.OutputFile(filename, header)
    # file.writePixels(data)
    # file.close()