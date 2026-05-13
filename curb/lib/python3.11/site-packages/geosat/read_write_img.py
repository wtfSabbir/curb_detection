import numpy as np


def read_img(filename, gdal=None):
    if gdal is None:
        try:
            from osgeo import ogr, gdal, osr
        except ImportError:
            import ogr
            import osr
            import gdal
    data = gdal.Open(filename)
    if not data:
        print('无法获取影像信息: ' + filename)
        exit(1)
    im_width = data.RasterXSize
    im_height = data.RasterYSize
    im_geotrans = data.GetGeoTransform()
    im_proj = data.GetProjection()
    im_data = data.ReadAsArray(0, 0, im_width, im_height)
    del data
    return im_proj, im_geotrans, im_data, [im_height, im_width]


def write_img(filename, im_data, im_proj=None, im_geotrans=None, int=None, no_data=None, gdal=None, save_format=None):
    if gdal is None:
        try:
            from osgeo import ogr, gdal, osr
        except ImportError:
            import ogr
            import osr
            import gdal
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    elif int:
        datatype = gdal.GDT_Byte
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    if save_format:
        driver = gdal.GetDriverByName("PNG")
    else:
        driver = gdal.GetDriverByName('GTiff')
    data = driver.Create(filename, im_width, im_height, im_bands, datatype)
    if no_data is not None:
        data.GetRasterBand(1).SetNoDataValue(no_data)
    if im_geotrans is not None:
        data.SetGeoTransform(im_geotrans)
    if im_proj is not None:
        data.SetProjection(im_proj)
    if im_bands == 1:
        data.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            data.GetRasterBand(i + 1).WriteArray(im_data[i])
    del data, im_data


# 拉伸图像 统一拉到8位
def stre_to_8(bands):
    out = np.zeros_like(bands, dtype=np.uint8)
    n = bands.shape[0]
    for i in range(n):
        t = stretching(bands[i, :, :], [0, 255])
        out[i, :, :] = t
    return out


def stretching(img, region):
    img = np.array(img, np.float32)
    max = np.percentile(img, 99.999)
    min = np.percentile(img, 0.001)
    img = ((img - min) / (max - min)) * (np.max(region) - np.min(region))
    img[img > np.max(region)] = np.max(region)
    img[img < np.min(region)] = np.min(region)
    return img
