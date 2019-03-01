repo = '/Users/jeremymiller/GoogleDrive/Data_Science/Projects/Baseball/'

with open(repo + "data/retrosheet_data/2018/2018SFN.EVN") as f:
    games = f.readlines()

# turn into dictionary format
# find way of keeping plays in sequence within an inning