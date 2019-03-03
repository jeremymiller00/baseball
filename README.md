# Baseball Project

The repository holds a set of baseball data projects. My goals are to anaylze baseball data, present visualizations to develop understanding of major league baseball, and explore alterative metrics for measuring player performance. The data come from [Retrosheet](https://www.retrosheet.org/), the [Sean Lahman Database](http://www.seanlahman.com/baseball-archive/statistics/), and  [Baseball Reference](https://www.baseball-reference.com/).  

My first task was to build a MongoDB database using the Retrosheet data. The Retrosheet event files contain a line for every play in a major league game, and well as game metadata. The files required parsing of each line to create a json format appropriate for MongoDB. This resulted is a flexible data base structure which lends itself to query and analysis.

The next step is to perfrom EDA on the data to get a greater understanding of the what the data contain, and to some cool plots!