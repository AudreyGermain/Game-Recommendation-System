from pandas import read_csv
import pathlib
import matplotlib.pyplot as plt

# get review info from csv
locationReviewFile = pathlib.Path(r'../../data/intermediate_data/steam_games_reviews.csv')
dataReviews = read_csv(locationReviewFile, usecols=["name", "percentage_positive_review"],)

plt.hist(x=dataReviews["percentage_positive_review"], range=[0, 100], bins=100)

# Add title and axis names
plt.title('Distribution of Game Reviews')
plt.ylabel('Number of Games')
plt.xlabel('Rating %')

plt.show()
