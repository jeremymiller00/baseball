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
docker exec -it mongoserver sh
```
This takes you to a shell environment.
To access the container with a command (go straight to the mongo shell):
```
docker exec -it mongoserver mongo
```

From here you can execute queries directy to a collection, but the real power for data analysis is in accessing the data via python.

