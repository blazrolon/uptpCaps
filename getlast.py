import os

def get_last_image():
    dir = 'pictures/'
    files = [os.path.join(dir, x) for x in os.listdir(dir) if x.endswith('.jpg')]
    newest = max(files, key = os.path.getctime)
    return newest