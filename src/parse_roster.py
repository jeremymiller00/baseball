'''Read in and parse lines from a directory of retrosheet data files; prep them for input into Mongodb.

Requires a directory titled "json_files" in the same parent directory as the diretories which contain the retrosheet event files.

Line formats:
{"entry_type": "roster", "player_id": "grimb101", "last_name": "Grimes", "first_name":"Burleigh", "bats": "R", "throws": "R", "team":"BRO", position": "P"}

in general, to import into mongo from command line
mongoimport --jsonArray --db testdb --collection testcoll < test.json

But here is what you should do:
rename all files in dir to prep for mongodb
ls | xargs -I {} mv {} c_{}

to import all in a folder, from inside the folder
ls -1 *.json | sed 's/.json$//' | while read col; do 
    mongoimport --jsonArray --db retrosheet --collection $col < $col.json; 
done

'''
import numpy as np
import sys
import json
import subprocess

def read_file(path):
    '''
    Read in a file.

    Parameters:
    ----------
    Input {str}: path to file
    Output: {str}: open python string object
    '''
    with open(path) as f:
        return f.readlines()

def trim_lines(file):
    '''
    Trim '\n' from the end of each line

    Parameters:
    ----------
    Input {str}: open python string object
    Output {str}: open python string object
    '''    
    out = []
    for line in file:
        out.append(line[:-1])
    return out

def split_lines(file):
    '''
    Split string into list of strings

    Parameters:
    ----------
    Input {str}: open python string object
    Output {list: str}: list of event strings
    '''
    out = []
    for line in file:
        out.append(line.split(","))
    return out

def build_dicts(file):
    '''
    Convert event string to dictionary with appropriate labels

    Parameters:
    ----------
    Input {list: str}: list of event strings
    Output {list: dict}: list of event dictionaries
    '''
    out = []
    for row in file:

        d = {"entry_type": "roster", "player_id":  row[0], "last_name": row[1], "first_name": row[2], "bats": row[3], "throws": row[4], "team": row[5], "position": row[6]}
        
        out.append(d)

    return out

def main(dir):
    '''
    Input: directory with retrosheet files
    Output: folder with json files
    '''
    files = subprocess.check_output(["ls", dir], universal_newlines=True)
    split_files = files.split("\n")
    counter = 0
    for file in split_files:
        if len(file) != 0:
            if file[-3:] == "ROS":
                counter += 1
                print("Processing file {} / {}".format(counter, len(split_files)))
                data = read_file(dir+file)
                trimmed_data = trim_lines(data)
                split_data = split_lines(trimmed_data)
                dicts = build_dicts(split_data)
                outpath = "data/retrosheet_data/json_files/"+file[:-3]+"json"
                with open(outpath, 'w') as fout:
                    json.dump(dicts, fout)

##################################################################
if __name__ == "__main__":

    '''
    ipython -i src/parse_retrosheet.py directory
    '''
    main(sys.argv[1])


