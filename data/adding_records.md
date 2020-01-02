# How to Add New Records to the Retrosheet Database

Make sure there is a directory on local in the format:
```
data/retrosheet_data/json_files
```

This directory is a holding zone for the json files before loading into mongodb.

Use the python command to parse the files into json:
```python
ipython -i src/parse_retrosheet.py directory_with_event_files/
```
Then rename all of the files in prep for mongodb by adding *_c* to the start of each filename:
```
ls | xargs -I {} mv {} c_{}
```
In the docker mongoserver container, the directory */home/data* is linked to *~* on local. From within the container, we need to navigate to the directory with the new json files.
```
docker exec -i mongoserver bash
cd /home/data/GoogleDrive/Data_Science/Projects/Baseball/data/retrosheet_data/json_files
```

Load the renamed json files into mongodb with:
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