# How to access the Retrosheet MongoDB Database with Docker

The retrosheet data has been loaded into a Mongo Database with parsing help from the python script in the src directory. 
```
docker container ls -a
```

To see all containers. The container with the retrosheet data is called **mongoserver**.

To start the container:
```
docker start mongoserver
```

To access the container:
```
docker exec -it mongoserver bash
```
This takes you to a shell environment.
To access the container with a command (go straight to the mongo shell):
```
docker exec -it mongoserver mongo
```

From here you can execute queries directy to a collection, but the real power for data analysis is in accessing the data via python.


Entry types from event files
```
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
```

Entry type from roster file
```
roster
{"entry_type": "roster", "player_id": "grimb101", "last_name": "Grimes", "first_name":"Burleigh", "bats": "R", "throws": "R", "team":"BRO", position": "P"}```

Entry type for teams file
Teams
```
{"entry_type": "teams", "abbreviation": "SFN", "league": "N", "city":"San Francisco", "name": "Giants"}```