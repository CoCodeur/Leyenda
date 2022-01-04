import shutil
import os 
import glob 


def preprocessing():
    path = './Dataset'
    types = ('*.png', '*.jpg')

    #SIZE OF DATASET
    painting_size = 3000
    schematic_size = 3000
    sketch_size = 3000
    text_size = 1000

    index_file = 0
    
    #Folder
    painting_folder = glob.glob(path+'/Painting/*.jpg')
    schematics_folder = glob.glob(path+'/Schematics/*.jpg')
    sketch_folder = glob.glob(path+'/Sketch/*.jpg')
    text_folder = glob.glob(path+'/Text/*.jpg')

    #Destination
    path = './Dataset/Other/'

    for i in range(3000):
        paint_file = str.split(painting_folder[i], '/')[-1]
        schematics_file = str.split(schematics_folder[i], '/')[-1]
        text_file = str.split(text_folder[i], '/')[-1]        


        shutil.move(painting_folder[i], path+ paint_file)
        shutil.move(schematics_folder[i], path+ schematics_file)
        shutil.move(text_folder[i], path+ text_file)

    for i in range(1000):
        sketch_file = str.split(sketch_folder[i], '/')[-1]
        shutil.move(sketch_folder[i], path+ sketch_file)

def main():

    preprocessing()


if __name__ == '__main__':
    main()
