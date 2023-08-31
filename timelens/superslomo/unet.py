

# import torch as th
# import torch.nn.functional as F
# from timelens.common import size_adapter
# from torch import nn
# import math



# def closest_larger_multiple_of_minimum_size(size, minimum_size):
#     return int(math.ceil(size / minimum_size) * minimum_size)

# class SizeAdapter(object):
#     """Converts size of input to standard size.
#     Practical deep network works only with input images
#     which height and width are multiples of a minimum size.
#     This class allows to pass to the network images of arbitrary
#     size, by padding the input to the closest multiple
#     and unpadding the network's output to the original size.
#     """

#     def __init__(self, minimum_size=64):
#         self._minimum_size = minimum_size
#         self._pixels_pad_to_width = None
#         self._pixels_pad_to_height = None

#     def _closest_larger_multiple_of_minimum_size(self, size):
#         return closest_larger_multiple_of_minimum_size(size, self._minimum_size)

#     def pad(self, network_input):
#         """Returns "network_input" paded with zeros to the "standard" size.
#         The "standard" size correspond to the height and width that
#         are closest multiples of "minimum_size". The method pads
#         height and width  and and saves padded values. These
#         values are then used by "unpad_output" method.
#         """
#         height, width = network_input.size()[-2:]
#         self._pixels_pad_to_height = (self._closest_larger_multiple_of_minimum_size(height) - height)
#         self._pixels_pad_to_width = (self._closest_larger_multiple_of_minimum_size(width) - width)
#         return nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(network_input)

#     def unpad(self, network_output):
#         """Returns "network_output" cropped to the original size.
#         The cropping is performed using values save by the "pad_input"
#         method.
#         """
#         return network_output[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]


# class up(nn.Module):
#     def __init__(self, inChannels, outChannels):
#         super(up, self).__init__()
#         self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

#     def forward(self, x, skpCn):
#         x = F.interpolate(x, scale_factor=2, mode="bilinear")
#         x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
#         x = F.leaky_relu(self.conv2(th.cat((x, skpCn), 1)), negative_slope=0.1)
#         return x


# class down(nn.Module):
#     def __init__(self, inChannels, outChannels, filterSize):
#         super(down, self).__init__()
#         self.conv1 = nn.Conv2d(
#             inChannels,
#             outChannels,
#             filterSize,
#             stride=1,
#             padding=int((filterSize - 1) / 2),
#         )
#         self.conv2 = nn.Conv2d(
#             outChannels,
#             outChannels,
#             filterSize,
#             stride=1,
#             padding=int((filterSize - 1) / 2),
#         )

#     def forward(self, x):
#         x = F.avg_pool2d(x, 2)
#         x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
#         x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
#         return x


# class UNet(nn.Module):
#     """Modified version of Unet from SuperSloMo.
    
#     Difference : 
#     1) there is an option to skip ReLU after the last convolution.
#     2) there is a size adapter module that makes sure that input of all sizes
#        can be processed correctly. It is necessary because original
#        UNet can process only inputs with spatial dimensions divisible by 32.
#     """

#     def __init__(self, inChannels, outChannels, ends_with_relu=True):
#         super(UNet, self).__init__()
#         self._ends_with_relu = ends_with_relu
#         self._size_adapter = size_adapter.SizeAdapter(minimum_size=32)
#         self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
#         self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
#         self.down1 = down(32, 64, 5)
#         self.down2 = down(64, 128, 3)
#         self.down3 = down(128, 256, 3)
#         self.down4 = down(256, 512, 3)
#         self.down5 = down(512, 512, 3)
#         self.up1 = up(512, 512)
#         self.up2 = up(512, 256)
#         self.up3 = up(256, 128)
#         self.up4 = up(128, 64)
#         self.up5 = up(64, 32)
#         self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

#     def forward(self, x):
#         # Size adapter spatially augments input to the size divisible by 32.
#         x = self._size_adapter.pad(x)
#         x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
#         s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
#         s2 = self.down1(s1)
#         s3 = self.down2(s2)
#         s4 = self.down3(s3)
#         s5 = self.down4(s4)
#         x = self.down5(s5)
#         x = self.up1(x, s5)
#         x = self.up2(x, s4)
#         x = self.up3(x, s3)
#         x = self.up4(x, s2)
#         x = self.up5(x, s1)

#         # Note that original code has relu et the end.
#         if self._ends_with_relu == True:
#             x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
#         else:
#             x = self.conv3(x)
#         # Size adapter crops the output to the original size.
#         x = self._size_adapter.unpad(x)
#         return x
