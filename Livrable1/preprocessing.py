
import os
import glob
import shutil


def main():

    folder_name = ['Painting',
                   'Photo',
                   'Schematics',
                   'Sketch',
                   'Text'
                   ]


    for name in folder_name:
        file_size = len(os.listdir('../Dataset/' + name + '/'))
        folder_name = "train"
        i = 0
        while len(os.listdir('../Dataset/' + name + '/')) != 0:
            if i >= int(( file_size * 80)/100):
                folder_name = "test"

            list_of_files = glob.glob('../Dataset/' + name + '/*')  # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            last_file_splitted = latest_file.split('/')
            shutil.move(latest_file, '../Dataset/' + folder_name + '/' + name + '/'+last_file_splitted[-1])
            print('../Dataset/' + folder_name + '/' + name + '/'+last_file_splitted[-1])
            i+=1



if __name__ == "__main__":
    main()
