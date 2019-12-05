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

### a. User Dataset

[//]: # (User Dataset description)
The first dataset is the [user](https://www.kaggle.com/tamber/steam-video-games) dataset. It contains the user id, the game, the behavior and the amount of hours played.
So each row of the dataset represent the behavior (play or purchase) of a user towards a game.
The amount of hours played is also specify, the column contains 1 if it's a purchase.
The dataset contains a total of 200000 rows, including 5155 different games and 12393 different users.

[//]: # (Reformat with purchase/play columns)
The third dataset is a list we split out the purchase/play column into two columns. Because The raw 'purchase' row records 1 hour, Clearly this doesn't make any sense so we remove it during the process of splitting the column and calculate the correct 'play' hours.

[//]: # (Histogram 1: all users: play + purchase)
Then we count the number of users for each games, and output a histograph to make the data visualization. 
| index |                                       game  |  user|       hrs|
| :-------------: | :------------- | -----: | -----: |
|  1336 |                                       Dota 2|   484|  981684.6|
|  4257 |                              Team Fortress 2|  2323|  173673.3|
|  4788 |                                     Unturned|  1563|   16096.4|
|   981 |             Counter-Strike Global Offensive |  1412|  322771.6|
|  2074 |                       Half-Life 2 Lost Coast|   981|     184.4|
|   984 |                        Counter-Strike Source|   978|   96075.5|
|  2475 |                                Left 4 Dead 2|   951|   33596.7|
|   978 |                               Counter-Strike|   856|  134261.1|
|  4899 |                                     Warframe|   847|   27074.6|
|  2071 |                       Half-Life 2 Deathmatch|   823|    3712.9|
|  1894 |                                  Garry's Mod|   731|   49725.3|
|  4364 |                   The Elder Scrolls V Skyrim|   716|   70616.3|
|  3562 |                                    Robocraft|   689|    9096.6|
|   980 | Counter-Strike Condition Zero Deleted Scenes|   679|     418.2|
|   979 |                Counter-Strike Condition Zero|   679|    7950.0|
|  2142 |                            Heroes & Generals|   658|    3299.5|
|  2070 |                                  Half-Life 2|   639|    4260.3|
|  3825 |                   Sid Meier's Civilization V|   596|   99821.3|
|  4885 |                                  War Thunder|   590|   14381.6|
|  3222 |                                       Portal|   588|    2282.8|
![Image text](https://github.com/AudreyGermain/Game-Recommendation-System/blob/master/plots/Histogram_AllUsersHrs.png?raw=true)

As you can see Dota 2 has the highest number of players and the highest number of total hours played so undeniably the most popular game. Where as other games such as "Half-Life 2 Lost Coast" have 981 users but a total of 184.4 hours played. I expect this game is in most cases a free bundle game. Some Games like these add noise to the dataset. So that's one of the reasons we use EM algorithms to create rating system for the games.

[//]: # (Histogram 2: only users: play)

[//]: # (Box plot)

### b. Game Dataset
[//]: # (Game Dataset description)
The second dataset contains a list of [games](https://www.kaggle.com/trolukovich/steam-games-complete-dataset/version/1) and their descriptions. It contains the url (directed to Steam store),
the type of package (app, bundle…), the name of the game, a short description, recent reviews, all reviews, release date,
developper, publisher, popular tags (Gore, Action, Shooter, PvP…), game detail (Multi-player, Single-player, Full controller support…),
languages, achievements, genre (Action, Adventure, RPG, Strategy…), game description, description of mature content,
minimum requirement to run the game, recommended requirement, original price and price with discount.
There is a total of 51920 games in the dataset.<br/>

## III. Methodology

We decided to use 3 differents algorithms to generate recommendation by user. We use 2 collaborative algorithm,
one using the ALS and one using the EM and SVD algorithms and we use one content-based algorithm.<br/>

### Collaborative Recommender

#### a. Training and Test Datasets
[//]: # (Describe splitting of user dataset into training and testing)
To create a training and testing dataset, we started by combining the information about play and purchase in a single row,
in this new form, the columns are user ID, name of the game, amount of hours of play time, play (0 if never played
and 1 if played) and purchase (technically always 1), this created a total of 128804 rows. Then we extracted 20% of
all the rows (25761 rows) for the test dataset and kept the rest  (103043 rows) for the training dataset.<br/>

#### b. Collaborative Recommender with ALS
This section describes a simple implementation of a collaborative filtering recommendation algorithm using matrix factorization with implicit data.
The work presented is based on the ["ALS Implicit Collaborative Filtering"](https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe "ALS Implicit Collaborative Filtering") and the ["A Gentle Introduction to Recommender Systems with Implicit Feedback"](https://jessesw.com/Rec-System/ "A Gentle Introduction to Recommender Systems with Implicit Feedback") blog posts.

Collaborative filtering does not require any information about the items or the users in order to provide recommendation. It only uses the interactions between users and items expressed by some kind of rating.

The data used for this recommender system is that of the steam users described in [subsection X](). The data does not explicitly contains the rating or preference of users towards the games, but rather it is **implicitly** expressed by the amount of hours users played games.

The Alternating Least Squares (ALS) is the model used to fit the data and generate recommendations. The ALS model is already implemented in the [implicit](https://github.com/benfred/implicit) python library thanks to [Ben Frederickson](http://www.benfrederickson.com/fast-implicit-matrix-factorization/).  
As described on its documentation [here](https://implicit.readthedocs.io/en/latest/als.html), the ALS algorithm available through the implicit library is a Recommendation Model based on the algorithms described in the paper [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) with performance optimizations described in [Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.6473&rep=rep1&type=pdf). The advantage of using the implicit python library versus a manual implementation of the algorithm is the speed required to generate recommendation since the ALS model in the implicit library uses Cython allowing the parallelization of the code among threads.

(Explain what's ALS and what it does)

In order to generate recommendations, the class ImplicitCollaborativeRecommender is implemented in a python script. The code is available here below.

[//]: # (<script src="https://gist.github.com/g30rdan/d992457bf34607493c19341c96761387.js"></script>)

(To complete)


#### c. Collaborative recommender with EM and SVD

Becaused our recommendation system should take consideration the games hasn't been played. We could create a rating system for games based on distribution of playing hours. Such like hours of some free bad games could have a distribution under 2 hours. As following, We use the EM algorithm rather than percentiles to present the distribution. In the EM algorithm, We use 5 groups as 5 stars to distinguish the good from the bad.

![Image text](https://raw.githubusercontent.com/AudreyGermain/Game-Recommendation-System/master/plots/EM%20plot.png?token=AKJKIZO5MDSWOLNZU7P6S2S55BWKU)

According to the plot, we could see the there are most of the users of Witcher 3 distribute in 4-5 groups. However there are a few users quickly lost their interests. It make sense to request a refund for the game that have benn played less than 2 hours. As you can see EM algorithm does a great job finding the groups of people with similar gaming habits and would potentially rate the game in a similar way. 

It does have some trouble converging which iThis example will use a gradient descent approach to find optimal U and V matrices which retain the actual observations with predict the missing values by drawing on the information between similar users and games. I have chosen a learning rate of 0.001 and will run for 200 iterations tracking the RMSE. The objective function is the squared error between the actual observed values and the predicted values. The U and V matrices are initialised with a random draw from a ~N(0, 0.01) distibution. This may take a few minutes to run.sn't surprising however the resulting distributions look very reasonable. 

A user-item matrix is created with the users being the rows and games being the columns. The missing values are set to zero. The observed values are the log hours for each observed user-game combination. The data was subset to games which have greater than 50 users and users which played the game for greater than 2 hours. This was chosen as 2 hours is the limit in which Steam will offer a return if you did not like the purchased game (a shout out to the Australian Competition and Consumer Commission for that one!).

The basic SVD approach will perform matrix factorisation using the first 60 leading components. Since the missing values are set to 0 the factorisation will try and recreate them which is not quite what we want. For this example we will simply impute the missing observations with a mean value.

This example will use a gradient descent approach to find optimal U and V matrices which retain the actual observations with predict the missing values by drawing on the information between similar users and games. I have chosen a learning rate of 0.001 and will run for 200 iterations tracking the RMSE. The objective function is the squared error between the actual observed values and the predicted values. The U and V matrices are initialised with a random draw from a ~N(0, 0.01) distibution. This may take a few minutes to run.


 
### Content-based Recommender

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
words in the chosen column (column_name) of each of the games, then it calculate the cosine similarity function.<br/>

```python
# Compute the Cosine Similarity matrix using the column
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(dataGames[column_name])
cosine_sim_matrix = cosine_similarity(count_matrix, count_matrix)
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
First, if the list of recommendation is empty (can happen if none of the game the user has are in the game dataset)
or not valid, a dataframe without recommendation is returned. If there is no problem with the recommendation list,
we get a dataframe of the name of the recommended games and the review (percentage of positive review) and we remove
the games that the user already has (no need to recommend a game the user already has). Then the recommendation are
ordered from the best review to the worst. If there is less recommendations then needed, empty spaces fill the rest of the column.
The reviews are used to order the recommendation since it's the easiest way to order them particularly considering that
not every games the user has produce recommendation because of the games in both datasets do not match totally like mentionned above.
If it wasn't because of that problem, we thought of taking into account the proportion of play time of each games to
recommend similar games to games that are most played. Using the reviews still ensure that the recommended games are
considered good in general by all the users.<br/>

All the dataframe rows produced by this functions are combine and are printed in a CSV file.<br/>

The whole script is arranged in a function to let us run it with multiple different input.
The different input of the content based recommender are generated in a script that rearrange the data in an easy 
to use it for the algorithm, it correspond to the varible column_name in the code above.<br/>

To prepare the data for the content based recommender we started by selecting the information we thought would be the 
most useful to find similar games. We read the useful column using the following code.

```python
dataGames = read_csv(locationGamesFile, usecols=["name", "genre", "game_details", "popular_tags", "publisher", "developer"])
```

Like mentioned previously, we decided to only keep the games that were both in the game dataset and the user dataset.
We chose to do it this way because there is no point in recommending games the user don't have and it will affect the 
performance of the algorithm compare to the other too. Also the dataset was too big to create matrix of cosine similarity
since it took to much memory. To match the games from both dataset together, we created an ID for the games by removing
all symbols the weren't alphanumeric, removing all capital letters and removing all spaces using to following code. 
We did the same on the games in the user dataset.

```python
# remove spaces and special character from game name in both dataset
for i, row in dataGames.iterrows():
    clean = re.sub('[^A-Za-z0-9]+', '', row["name"])
    clean = clean.lower()
    dataGames.at[i, 'ID'] = clean
```
After this, we found all the uniques ID from the user dataset and kept only the rows in the games dataset where the ID 
matched one of the ID in the user dataset. This way we were able to get 3036 games of the game dataset that matched 
some of the 5151 games from the user dataset. Without the ID we only were able to find 71 games that matched only 
using the names. Since we have less games in the new game dataset, the recommender system will not be able to find 
recommendation for every games in the user dataset. This will surly affect its performance.<br/>

With the new smaller game dataset, we made sure to remove the spaces from le useful column we chose to use.
By removing the spaces, we ensure that, for exemple, Steam Achievement and Steam Cloud dont get a match because they 
both contain Steam. Therefor we apply the following function to all the columns like this.

```python
def clean_data(x):
    if isinstance(x, str):
        return x.replace(" ", "")
    else:
        print(x)
        return x


usedGames.loc[:, 'genre'] = usedGames['genre'].apply(clean_data)
```

Finally, we created some custom columns by combining multiple column to try and find the combination of information 
that will give us the best recommendation system possible. 

```python
# create some column containing a mix of different information
usedGames["genre_publisher_developer"] = usedGames['genre'] + usedGames['publisher'] + usedGames['developer']
usedGames["genre_popular_tags_developer"] = usedGames['genre'] + usedGames['popular_tags'] + usedGames['developer']
usedGames["genre_popular_tags_game_details"] = usedGames['genre'] + usedGames['popular_tags'] + usedGames['game_details']
usedGames["genre_publisher_developer_game_details"] = usedGames['genre'] + usedGames['publisher'] + usedGames['developer'] + usedGames['game_details']
```

The results of the different columns will be compared in the Evaluation & Analysis section of this article.
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
