from  PIL import Image
import os    
import glob

def extract_predict_REAL(test_images):
    # Opens a image in RGB mode
    im = Image.open(test_images)
    # Size of the image in pixels (size of original image)
    width, height = im.size
    # Setting the points for cropped image
    left = 0
    top = 0
    right = 256*1-1
    bottom = height
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))
    im1 = im1.resize((167, 167))
    return im1

def extract_predict_GEN(test_images):
    # Opens a image in RGB mode
    im = Image.open(test_images)
    # Size of the image in pixels (size of original image)
    width, height = im.size
    # Setting the points for cropped image
    left = 256*3
    top = 0
    right = 256*4-1
    bottom = height
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))
    im1 = im1.resize((167, 167))
    return im1



if __name__ == '__main__':
    list_files = sorted(glob.glob(os.path.abspath('DC2Anet_db/test/20221204-0200/*.jpg')),  key=len)
    index = 0
    output_file_dir = 'dataset/test_predict/CT_REAL/'
    for image_address in list_files:       
        # for PT address
        output_file = output_file_dir + str(index) + ".jpg"
        index += 1
        extract_predict_REAL(image_address).save(output_file)
    
    list_files = sorted(glob.glob(os.path.abspath('DC2Anet_db/test/20221204-0200/*.jpg')),  key=len)
    index = 0
    output_file_dir = 'dataset/test_predict/CT_GEN/'
    for image_address in list_files:       
        # for PT address
        output_file = output_file_dir + str(index) + ".jpg"
        index += 1
        extract_predict_GEN(image_address).save(output_file)


def extract_predict(test_images):
    # Opens a image in RGB mode
    im = Image.open(test_images)
    # Size of the image in pixels (size of original image)
    width, height = im.size
    # Setting the points for cropped image
    left = 256*2
    top = 0
    right = 256*3-1
    bottom = height
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))
    im1 = im1.resize((167, 167))
    return im1