import numpy as np 

repo = '/Users/jeremymiller/GoogleDrive/Data_Science/Projects/Baseball/'

with open(repo + "data/retrosheet_data/2018/2018SFN.EVN") as f:
    games = f.readlines()

# get rid of \n
games1 = []
for line in games:
    games1.append(line[:-2])

# put game number in front of each line
games2 = []
id_field = ""
for line in games1:
    if line[0:2] == "id":
        id_field = line
        games2.append(id_field)
    else:
        games2.append(",".join([id_field, line]))

# break each game up by game id
games3 = []
for line in games2:
    games3.append(line.split(","))

games3[0]

outlist = []
for row in games3:
    if len(row) < 3:
        continue # eliminate original id rows
    if row[2] == "info":
        if len(row) > 4: # make sure it's not an empty "save" row
        # outlist.append(row)
            d = {"game_id": row[1], "entry_type": row[2], row[3]: row[4]}
            outlist.append(d)
    elif row[2] == "start":
        continue
    elif row[2] == "play":
        continue
    
    elif row[2] == "com":
        continue
    
    elif row[2] == "sub":
        continue
    
    elif row[2] == "data":
        continue

outlist
games3[:100]

# find way of keeping plays in sequence within an inning

'''
Mongo Sketch:
info
{"game_id": SFN201804100, "entry_type": "info", "visteam":"ARI}

start
{"game_id": SFN201804100, "entry_type": "start", player_id: "dysoj001", "name": "Jarrod Dyson", "team": 0, "batting_order": 1, "position" : 7}

play
{"game_id": SFN201804100, "entry_type": "play", "inning": 1, "bottom": 0, "player_id": "dysoj001", "count": 21, "pitches", CBBX, "event": "CBF2FBB>X,D7/L.2-H;1-H"

com
{"game_id": SFN201804100, "entry_type": "com", "comment": "team blah blah" 

sub
{"game_id": SFN201804100, "entry_type": "sub", player_id: "dysoj001", "name": "Jarrod Dyson", "team": 0, "batting_order": 1, "position" : 7}

data
{"game_id": SFN201804100, "entry_type": "data", "data_type": "er", "player_id": "kersc001", "earned_runs": 1}

'''