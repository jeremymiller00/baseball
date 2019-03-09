# Baseball Project

The repository holds a set of baseball data projects. My goals are to anaylze baseball data, present visualizations to develop understanding of major league baseball, and explore alterative metrics for measuring player performance. The data come from [Retrosheet](https://www.retrosheet.org/), the [Sean Lahman Database](http://www.seanlahman.com/baseball-archive/statistics/), and  [Baseball Reference](https://www.baseball-reference.com/).  

My first task was to build a MongoDB database using the Retrosheet data. The Retrosheet event files contain a line for every play in a major league game, and well as game metadata. The files required parsing of each line to create a json format appropriate for MongoDB. This resulted is a flexible data base structure which lends itself to query and analysis.

The next step is to perfrom EDA on the data to get a greater understanding of the what the data contain, and to some cool plots!

## Motivation:

Baseball statistics are typically analyzed and presented as **summaries**, either of a game, season, player, or some other entity. My analysis begins from the observation that baseball is a game of **context**. The same outcome on a given play can have radically different meanings for the outcome of a game depending on the context. I will seek to analyze the game based on this contextual understanding. I am not the first to observe that at the end of baseball season there is one number that matter far more than any other: wins, specifically number of wins less than the team's division leader. Obviously this mirrors the situation in any one game, where the only number that matters at the end of the game is the number of runs relative to the other team. Based on these dynamics, a given outcome of an at-bat (fly ball to center field, ground ball to second base) could be viewed as a success, a failure, or something of a mixture. This situational complexity is, for me, on the reasons I enjoy baseball. 

## Project Ideas

* Emperically define baseball "eras". This is often done through observation, but how much would these definitions differ if eras were defined numerically?

* Context-informed player / team statistics. This is difficult. The basic idea is this: in a given at-bat, given the game context, did the player (pitcher, hitter, fielder, or runner) achieve their *primary objective*? Some assumptions need to be made here, but they all stem from the core assumption that a player's main motivation is to help his team win today's game.

* Comparing offense vs. defense (including pitching): does one have a greater influence on wins overall?

* Rank a game's excitement level for fans who want to go back and watch historical games. 

* Explore 19th century baseball. This project is motivated purely by my own curiosity. It can follow from the first project idea: once eras are empirically defined, what are the characteristics of each era? I find it fun to imagine going to a baseball game in those early days.




