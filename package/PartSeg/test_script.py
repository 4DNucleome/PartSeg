from PartSegImage import ImageReader

img = ImageReader().read("/mnt/raid/rosliny/2019-07-08/rsr150_Series001_decon.tif")
print(img.spacing)