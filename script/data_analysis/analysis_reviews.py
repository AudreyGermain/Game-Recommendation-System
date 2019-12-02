from pandas import read_csv
import re
import pathlib

# Get data from CSV
locationGamesFile = pathlib.Path(r'../../data/raw_data/steam_games.csv')
dataGames = read_csv(locationGamesFile)

# add and initialize new columns
dataGames["review_qualification"] = ""
dataGames["percentage_positive_review"] = -1

for i, row in dataGames.iterrows():
    if type(row["all_reviews"]) == str:

        # extract % of positive reviews
        x = re.findall(r'- [0,1,2,3,4,5,6,7,8,9]*%', row["all_reviews"])
        if len(x) != 0:
            dataGames.at[i, 'percentage_positive_review'] = x[0].translate({ord(i): None for i in '- %'})

        # extract qualification of reviews
        reviewParse = row["all_reviews"].split(",")
        if 'user reviews' in reviewParse[0]:
            dataGames.at[i, 'review_qualification'] = ""
        else:
            dataGames.at[i, 'review_qualification'] = reviewParse[0]

# list of possible review qualification
possibleReview = dataGames["review_qualification"].unique()
print(possibleReview)

# print csv of reviews
dataGames.to_csv(pathlib.Path(r'../../data/intermediate_data/steam_games_reviews.csv'),
                 columns=["name", "percentage_positive_review", "review_qualification", "all_reviews"],
                 index=False)
