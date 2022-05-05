
# return
def key(x):
    return x[1]

# return class of image
def get_class(path):
    length = len(path.split('/'))
    class_of_img = path.split('/')[length-2]
    return class_of_img
