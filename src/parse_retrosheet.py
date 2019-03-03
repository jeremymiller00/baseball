'''Read in and parse lines from a directory of retrosheet data files; prep them for input into Mongodb.

Requires a directory titled "json_files" in the same parent directory as the diretories which contain the retrosheet event files.

Line formats:
info
{"game_id": SFN201804100, "entry_type": "info", "visteam":"ARI}

start
{"game_id": SFN201804100, "entry_type": "start", player_id: "dysoj001", "name": "Jarrod Dyson", "home_team": 0, "batting_order": 1, "position" : 7}

play
{"game_id": SFN201804100, "entry_type": "play", "inning": 1, "bottom": 0, "player_id": "dysoj001", "count": 21, "pitches", CBBX, "event": "CBF2FBB>X,D7/L.2-H;1-H"

com
{"game_id": SFN201804100, "entry_type": "com", "comment": "team blah blah" 

sub
{"game_id": SFN201804100, "entry_type": "sub", player_id: "dysoj001", "name": "Jarrod Dyson", "team": 0, "batting_order": 1, "position" : 7}

data
{"game_id": SFN201804100, "entry_type": "data", "data_type": "er", "player_id": "kersc001", "earned_runs": 1}

to import into mongo from command line
mongoimport --jsonArray --db testdb --collection testcoll < test.json

rename all files in dir to prep for mongodb
ls | xargs -i mv {} c_{}

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
    Input: 
    Output: 
    '''
    with open(path) as f:
        return f.readlines()

def trim_lines(file):
    # get rid of \n
    '''
    Read in a file.

    Parameters:
    ----------
    Input: 
    Output: 
    '''    
    out = []
    for line in file:
        out.append(line[:-1])
    return out

def insert_game_number(file):
    # put game number in front of each line
    '''
    Read in a file.

    Parameters:
    ----------
    Input: 
    Output: 
    '''
    out = []
    id_field = ""
    for line in file:
        if line[0:2] == "id":
            id_field = line
            out.append(id_field)
        else:
            out.append(",".join([id_field, line]))
    return out

def split_lines(file):
    # split lines
    '''
    Read in a file.

    Parameters:
    ----------
    Input: 
    Output: 
    '''
    out = []
    for line in file:
        out.append(line.split(","))
    return out

def build_dicts(file):
    # build dictionaries
    '''
    Read in a file.

    Parameters:
    ----------
    Input: 
    Output: 
    '''
    out = []
    for row in file:
        if len(row) < 3:
            continue # eliminate original id rows

        if row[2] == "info":
            if len(row) > 4: # make sure it's not an empty "save" row
            # outlist.append(row)
                d = {"game_id": row[1], "entry_type": row[2], row[3]: row[4]}
                out.append(d)

        elif row[2] == "start":
            d = {"game_id": row[1], "entry_type": row[2], "player_id":  row[3], "name": row[4], "home_team": row[5],     "batting_order": row[6], "position": row[7]}
            out.append(d)

        elif row[2] == "play":
            d = {"game_id": row[1], "entry_type": row[2],"inning": row[3], "bottom": row[4], "player_id": row[5], "count": row[6], "pitches": row[7], "event": row[8]}
            out.append(d)

        elif row[2] == "com":
            d = {"game_id": row[1], "entry_type": row[2], "comment": row[3]}
            out.append(d)

        elif row[2] == "sub":
            d = {"game_id": row[1], "entry_type": row[2], "player_id": row[3], "name": row[4], "home_team": row[5], "batting_order": row[6], "position": row[7]}
            out.append(d)

        elif row[2] == "data":
            d = {"game_id": row[1], "entry_type": row[2], "data_type": row[3], "player_id": row[4], "earned_runs": row[5]}

    return out

def main(dir):
    '''
    Input: directory with retrosheet files
    Output: folder with json files
    '''
    files = subprocess.check_output(["ls", dir], universal_newlines=True)
    split_files = files.split("\n")
    for file in split_files:
        if len(file) != 0:
            if file[-3] == "E":
                data = read_file(dir+file)
                trimmed_data = trim_lines(data)
                id_data = insert_game_number(trimmed_data)
                split_data = split_lines(id_data)
                dicts = build_dicts(split_data)
                outpath = "data/retrosheet_data/json_files/"+file[:-3]+"json"
                with open(outpath, 'w') as fout:
                    json.dump(dicts, fout)

##################################################################
if __name__ == "__main__":

    '''
    ipython -i parse_retrosheet.py directory
    '''
    main(sys.argv[1])


