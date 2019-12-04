# Recommendation System for Steam Game Store: An overview of recommender systems

## Team:<br/>
- Doo Hyodan, Department Of Information System, Hanyang University, ammissyouyou@gmail.com<br/>
- Audrey Germain, Computer Engineering, Department of Software Engineering and Computer Engineering, Polytechnique Montreal<br/>
- Geordan Jove, Aerospace Engineering, Department of Mechanical Engineering, Polytechnique Montreal<br/>

## I. Introduction
Like many young people, all member of this team have an interest in video games, more particularly in computer games.
Therefore, this project has for goal to build a recommender system for computer game.<br/>

This project was implemented as our final project for the course Introduction to Artificial Intelligence (ITE351) at Hanyang University.<br/>

For this project we are using the data from Steam the biggest video game digital distribution service for computer games.
We will be using some user data and game data, the datasets used will be explained in detail further in this blog.<br/>

The primary focus of this project is to build a recommendation system that will recommend games for each user based on
their preferences and gaming habits. In order to make our recommendation system the best possible, multiple algorithms
will be implemented and compared to make the recommendation the most relevant possible for each users.<br/>

There are three main types of recommendation system: collaborative filtering, content-based filtering and hybrid recommendation system. 
The collaborative filtering is based on the principle that if two people liked the same things in the past, 
if one of them like something the other is likely to like it too. 
The advantage of this filtering method is that the algorithm doesn’t need to understand or process the content 
of the items it recommends. The content-based filtering is based on the description of the items to recommend 
similar items and recommend items similar to what a user likes. 
Finally, the hybrid recommendation system consists of combining content based and collaborative filtering, 
either by using an algorithm that uses both or by combining the recommendation found by both methods. 
According to research combining both results is a better recommendation than using only one of them. <br/>

The data used for a recommendation system can be explicit, such as comment or rating, 
or implicit, such as behavior and events like order history, search logs, clicks, etc. 
The implicit data is harder to process because it’s hard to determine which information is useful and useless, 
but it’s easier to acquire than explicit data because the user doesn’t need to do anything more than 
using the website or app as usual. <br/>

In this project we implemented 2 collaboratives and 1 content based algorithm. 
We used implicite data from the users to implement them.

## II. Datasets
For this project, 2 differents datasets are used. Both dataset are available for free on kaggle and are extracted from Steam.<br/>

The first dataset is the [user](https://www.kaggle.com/tamber/steam-video-games) dataset. It contains the user id, the game, the behavior and the amount of hours played.
So each row of the dataset represent the behavior (play or purchase) of a user towards a game.
The amount of hours played is also specify, the column contains 1 if it's a purchase.
The dataset contains a total of 200000 rows, including 5155 different games and 12393 different users.
To create a training and testing dataset, we started by combining the information about play and purchase in a single row,
in this new form, the columns are user ID, name of the game, amount of hours of play time, play (0 if never played
and 1 if played) and purchase (technically always 1), this created a total of 128804 rows. Then we extracted 20% of
all the rows (25761 rows) for the test dataset and kept the rest  (103043 rows) for the training dataset.<br/>

The second dataset contains a list of [games](https://www.kaggle.com/trolukovich/steam-games-complete-dataset/version/1) and their descriptions. It contains the url (directed to Steam store),
the type of package (app, bundle…), the name of the game, a short description, recent reviews, all reviews, release date,
developper, publisher, popular tags (Gore, Action, Shooter, PvP…), game detail (Multi-player, Single-player, Full controller support…),
languages, achievements, genre (Action, Adventure, RPG, Strategy…), game description, description of mature content,
minimum requirement to run the game, recommended requirement, original price and price with discount.
There is a total of 51920 games in the dataset.<br/>

## III. Methodology

We decided to use 3 differents algorithms to generate recommendation by user. We use 2 collaborative algorithm,
one using the ALS and one using the EM and SVD algorithms and we use one content-based algorithm.<br/>

### Collaborative recommender with ALS
This section describes a simple implementation of a collaborative filtering recommendation algorithm using matrix factorization with implicit data.
The work presented is based on the ["ALS Implicit Collaborative Filtering"](https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe "ALS Implicit Collaborative Filtering") and the ["A Gentle Introduction to Recommender Systems with Implicit Feedback"](https://jessesw.com/Rec-System/ "A Gentle Introduction to Recommender Systems with Implicit Feedback") blog posts.

Collaborative filtering does not require any information about the items or the users in order to provide recommendation. It only uses the interactions between users and items expressed by some kind of rating.

The data used for this recommender system is that of the steam users described in [subsection X](). The data does not explicitly contains the rating or preference of users towards the games, but rather it is **implicitly** expressed by the amount of hours users played games.

The Alternating Least Squares (ALS) is the model used to fit the data and generate recommendations. The ALS model is already implemented in the [implicit](https://github.com/benfred/implicit) python library thanks to [Ben Frederickson](http://www.benfrederickson.com/fast-implicit-matrix-factorization/).  
As described on its documentation [here](https://implicit.readthedocs.io/en/latest/als.html), the ALS algorithm available through the implicit library is a Recommendation Model based on the algorithms described in the paper [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) with performance optimizations described in [Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.6473&rep=rep1&type=pdf). The advantage of using the implicit python library versus a manual implementation of the algorithm is the speed required to generate recommendation since the ALS model in the implicit library uses Cython allowing the parallelization of the code among threads.

(Explain what's ALS and what it does)

In order to generate recommendations, the class ImplicitCollaborativeRecommender is implemented in a python script. The code is available here below.

<script src="https://gist.github.com/g30rdan/d992457bf34607493c19341c96761387.js"></script>


### Collaborative recommender with EM and SVD

Becaused our recommendation system should take consideration the games hasn't been played. We could create a rating system for games based on distribution of playing hours. Such like hours of some free bad games could have a distribution under 2 hours. As following, We use the EM algorithm rather than percentiles to present the distribution. In the EM algorithm, We use 5 groups as 5 stars to distinguish the good from the bad.

![Image text](https://raw.githubusercontent.com/AudreyGermain/Game-Recommendation-System/master/plots/EM%20plot.png?token=AKJKIZO5MDSWOLNZU7P6S2S55BWKU)

According to the plot, we could see the there are most of the users of Witcher 3 distribute in 4-5 groups. However there are a few users quickly lost their interests. It make sense to request a refund for the game that have benn played less than 2 hours. As you can see EM algorithm does a great job finding the groups of people with similar gaming habits and would potentially rate the game in a similar way. It does have some trouble converging which isn't surprising however the resulting distributions look very reasonable. 


 
### Content-based recommender

To generate the recommendation for each game, the following function is used. The input of the function is the title of
the game as a string and the cosine matrix (explained later) and the output is a list of recommended game title.<br/>

```python
def get_recommendations(title, cosine_sim):

	if title not in listGames:
    	return []

	# Get the index of the game that matches the name
	idx = indices[title]

	# if there's 2 games or more with same name (game RUSH)
	if type(idx) is Series:
    	return []

	# Get the pairwise similarity scores of all games with that game
	sim_scores = list(enumerate(cosine_sim[idx]))

	# Sort the games based on the similarity scores
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

	# Get the scores of the most similar games
	# (not the first one because this games as a score of 1 (perfect score) similarity with itself)
	sim_scores = sim_scores[1:n_recommendation + 1]

	# Get the games indices
	movie_indices = [i[0] for i in sim_scores]

	# Return the top most similar games
	return dataGames['name'].iloc[movie_indices].tolist()

```
The variable listGames is a list of all the games that are in both of the dataset (user and game dataset).
We use this because there is a lot of games in the game dataset that have never been played or purchased by any user,
so there's no use in considering them in the recommender and some of the games in the user dataset are not in the game dataset.
To maximize the amount of match between the game titles in the datasets we removed all symboles and spaces and put
every letters in lower case. We were able to find 3036 games in the game dataset that match some of the 5151 games that are in the user dataset.<br/>

The variable indices is a reverse map that use the name as key to get the index of each game in the cosine similarity matrix.
We make sure that the idx is not a Series, it can happen in the case where 2 different games have the same name (in our dataset 2 games have the name "RUSH").<br/>
```python
# Construct a reverse map of indices and game names
indices = Series(dataGames.index, index=dataGames['name']).drop_duplicates()
```
Then we get the similarity score of each games from the matrix and we order it from the most similar to the less similar.
Finally we just need to extract the amount of recommendation that we want and return the list.
The variable n_recommendation contains the amount of recommendation we want to get, we decided to set it to 20.<br/>

To generate the cosine similarity matrix we use the following code. First it calculate the matrix of frequency of each
words in the popular tag of each of the games, then it calculate the cosine similarity function.<br/>

```python
# Compute the Cosine Similarity matrix using the popular tags column
count = CountVectorizer(stop_words='english')
count_matrix_popular_tags = count.fit_transform(dataGames['popular_tags'])
cosine_sim_matrix_popular_tags = cosine_similarity(count_matrix_popular_tags, count_matrix_popular_tags)
```
To get the recommendation for each user, we implemented a function that combines the recommendations and get the
recommendation with the best reviews (extracted from the game dataset). This function takes the ID of each user,
the list of recommendation (the recommendation function explained previously is applied to all the games a user
has and a list of all the recommendations is made) and the list of all the game the user already has. The function
return a dataframe containing the user ID in the first column and then 20 column with the top recommendations.<br/>

```python
def make_recommendation_for_user(user_id, game_list, game_user_have):
	if type(game_list) is not list or len(game_list) == 0:
    	# return empty one
    	return DataFrame(data=[[user_id] + [""] * n_recommendation], columns=col_names)

	# get reviews of game recommendation, remove the games the user already has and order them by reviews
	recommendation_reviews = dataReviews.loc[dataReviews['name'].isin(game_list)]
	recommendation_reviews = recommendation_reviews.loc[~recommendation_reviews['name'].isin(game_user_have)]
	recommendation_reviews = recommendation_reviews.sort_values(by="percentage_positive_review", ascending=False)

	if len(recommendation_reviews.index) < n_recommendation:
    	return DataFrame(data=[[user_id] + recommendation_reviews["name"].tolist() +
                           	[""] * (n_recommendation - len(recommendation_reviews.index))],
                     	columns=col_names)
	else:
    	return DataFrame(data=[[user_id] + recommendation_reviews["name"].tolist()[0:n_recommendation]],
                     	columns=col_names)
```
First, is the list of recommendation is empty (can happen if none of the game the user has are in the game dataset)
or not valid, a dataframe without recommendation is returned. If there is no problem with the recommendation list,
we get a dataframe of the name of the recommended games and the review (percentage of positive review) and we remove
the games that the user already has (no need to recommend a game the user already has). Then the recommendation are
ordered from the best review to the worst. If there is less recommendations then needed, empty spaces fill the rest of the column.<br/>

All the dataframe rows produced by this functions are combine and are printed in a CSV file.<br/>

## IV. Evaluation & Analysis

To compare the different algorithms we created a script that calculate the ratio of games the user has 
(in the test dataset) that are in the top 20 recommendations and the amount of games the user has in the
test dataset. The mean of the ratio for all users is the calculated. The ratio is a bit low because for some
user in the training dataset it was not possible to get the recommendations and some user don't have games
in the test dataset, in those cases the ratio will be 0. <br>

First of all, we compared the content based algorithm with different input. 
Those input are either column from the original dataset or a combination of different columns. 
In the following table we ca see the ratio for the different inputs. 
As we can see, the best algorithm is the one that uses the genre, publisher and developer of the game as input.
This version will be used to compare the other 2 algorithms.

Algorithm| Ratio
------------ | -------------
Popular tags| 0.6455%
Genre | 1.1295%
Genre, popular tags & developer | 0.6992%
Genre, popular tags & game details | 0.9234%
Genre, publisher & developer | 1.8377%
Genre, publisher, developer & game details | 1.6943%

We calculated the ratio in the same way for both of the collaborative algorithm, the result are in the following table.
As we can see, the collaborative recommender with ALS is the best one, followed by the content based recommender.
The performance of the collaborative recommendation system with EM and SVD is far behind the other 2.

Algorithm| Ratio
------------ | -------------
Collaborative with ALS| 2.6707%
Collaborative with EM and SVD | 0.2557%
Content-based (Genre, publisher & developer) | 1.8377%

## V. Related Work

To understand what a recommender system is and the different types, we used [this article](https://marutitech.com/recommendation-engine-benefits/).<br/>

To manipulate the datasets, we used the [pandas](https://pandas.pydata.org/) library.<br/>

For the content-based recommender system we used the some code from the blog post
[Recommender Systems in Python: Beginner Tutorial](https://www.datacamp.com/community/tutorials/recommender-systems-python?fbclid=IwAR1fz9YLOgZ95KHwoLpgb-hTdV2MekujDGBngRTG3kYmBJYxwSK3UWvNJDg)
to implement the function that give the recommendation for each games.<br/>

For the EM algorithm we used the article
[Machine Learning with Python: Exception Maximization and Gaussian Mixture Models in Python](https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php).<br/>

## VI. Conclusion: Discussion

It would be interesting to create an hybrid recommender system using the collaborative recommender with the ALS 
algorithm and the content based algorithm using the genre, publisher and developer as input to see if we can make 
a recommender system even more effective then the 2 on their own.
