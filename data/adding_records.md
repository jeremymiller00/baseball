# How to Add New Records to the Retrosheet Database
## These are instructions for adding event files to the database.

Temporarily stop any auto-sync. Restart afterwards. This will avoid expesive and unnecessary backup of temp files.  

Make sure there is a directory on local in the format:
```
data/retrosheet_data/json_files
```

This directory is a holding zone for the json files before loading into mongodb.

Use the python command to parse the event files into json:
```python
ipython -i src/parse_events.py directory_with_event_files/
```

In the docker mongoserver container, the directory */home/* is linked to */Users/Jeremy/GoogleDrive/Data_Science/Projects/Baseball* on local. From within the container, we need to navigate to the directory with the new json files.
```
docker exec -it mongoserver bash
cd /home/data/data/retrosheet_data/json_files
```

To load all into single collection (named 'logs'):
```
ls -1 *.json | sed 's/.json$//' | while read col; do 
    mongoimport --jsonArray --db retrosheet --collection logs < $col.json;
done
```

Load the renamed json files into mongodb each with it's own colllection with:
```
ls -1 *.json | sed 's/.json$//' | while read col; do 
    mongoimport --jsonArray --db retrosheet --collection $col < $col.json;
done
```

Finally, from back in the terminal, delete the *json_files* directory to clean up what is no longer needed.
```
rm -rf data/retrosheet_data/json_files/
```

### And there you have it!


Old steps:
Then rename all of the files in prep for mongodb by adding *_c* to the start of each filename:
```
ls | xargs -I {} mv {} c_{}
```