from math import sqrt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
global uid
global mname
import tkinter as tk
from PIL import Image, ImageTk


root = tk.Tk()
#root.geometry("1400x500")
root.title("Movie Recommandation System")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Home Page")

   
mname = tk.StringVar()
uid=tk.IntVar()
user_id=tk.IntVar()


#####For background Image
image2 = Image.open('9.jpeg')
image2 = image2.resize((w,h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)


#
label_l2 = tk.Label(root, text="___Movie Recommendation System___",font=("times", 30, 'bold'),
                    background="black", fg="white", width=68, height=2)
label_l2.place(x=0, y=0)

# Reading ratings file
ratings = pd.read_csv('D:/Movie-Recommendation-system/ratings.csv', sep=',', encoding='latin-1', usecols=['userId','movieId','rating','timestamp'])

# Reading movies file
movies = pd.read_csv('D:/Movie-Recommendation-system/movies.csv', sep=',', encoding='latin-1', usecols=['movieId','title','genres'])
df_movies = movies 
df_ratings = ratings 
print(df_ratings)
print(df_movies)
#Exploratory Data Analysis(EDA)
df_movies.head(5)


plt.figure(figsize=(20,7))
generlist = df_movies['genres'].apply(lambda generlist_movie : str(generlist_movie).split("|"))
geners_count = {}

# #Most popular genres of movie released
# for generlist_movie in generlist:
#     for gener in generlist_movie:
#         if(geners_count.get(gener,False)):
#             geners_count[gener]=geners_count[gener]+1
#         else:
#             geners_count[gener] = 1       
# geners_count.pop("(no genres listed)")
# plt.bar(geners_count.keys(),geners_count.values(),color='m')

df_ratings.head(5)
#Distribution of users rating

sns.distplot(df_ratings["rating"]);

print("Shape of frames: \n"+ " Rating DataFrame"+ str(df_ratings.shape)+"\n Movies DataFrame"+ str(df_movies.shape))
merge_ratings_movies = pd.merge(df_movies, df_ratings, on='movieId', how='inner')
merge_ratings_movies.head(2)
print(merge_ratings_movies)
merge_ratings_movies = merge_ratings_movies.drop('timestamp', axis=1)
merge_ratings_movies.shape
print(merge_ratings_movies)
ratings_grouped_by_users = merge_ratings_movies.groupby('userId').agg([np.size, np.mean])
ratings_grouped_by_users.head(2)
print(ratings_grouped_by_users)

ratings_grouped_by_users = merge_ratings_movies.groupby('userId').agg([np.size, np.mean])
ratings_grouped_by_users.head(2)
print(ratings_grouped_by_users)
ratings_grouped_by_users = ratings_grouped_by_users.drop('movieId', axis = 1)
print(ratings_grouped_by_users)
#ratings_grouped_by_users['rating']['size'].sort_values(ascending=False).head(10).plot('bar', figsize = (10,5))

ratings_grouped_by_movies = merge_ratings_movies.groupby('movieId').agg([np.mean], np.size)
ratings_grouped_by_movies.shape
print(ratings_grouped_by_movies)
ratings_grouped_by_movies.head(3)
print(ratings_grouped_by_movies)
ratings_grouped_by_movies = ratings_grouped_by_movies.drop('userId', axis=1)
print(ratings_grouped_by_movies)
ratings_grouped_by_movies['rating']['mean'].sort_values(ascending=False).head(10).plot(kind='barh', figsize=(7,6));
low_rated_movies_filter = ratings_grouped_by_movies['rating']['mean']< 1.5
low_rated_movies = ratings_grouped_by_movies[low_rated_movies_filter]
low_rated_movies.head(20).plot(kind='barh', figsize=(7,5));

low_rated_movies.head(10)

# ##########content based ###########




# Define a TF-IDF Vectorizer Object.
tfidf_movies_genres = TfidfVectorizer(token_pattern = '[a-zA-Z0-9\-]+')
#print(tfidf_movies_genres)

#Replace NaN with an empty string
df_movies['genres'] = df_movies['genres'].replace(to_replace="(no genres listed)", value="")


#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_movies_genres_matrix = tfidf_movies_genres.fit_transform(df_movies['genres'])
# print(tfidf_movies_genres.get_feature_names())
# Compute the cosine similarity matrix
# print(tfidf_movies_genres_matrix.shape)
# print(tfidf_movies_genres_matrix.dtype)
cosine_sim_movies = linear_kernel(tfidf_movies_genres_matrix, tfidf_movies_genres_matrix)
cosine_sim_movies1 = linear_kernel(tfidf_movies_genres_matrix, tfidf_movies_genres_matrix)
# print(cosine_sim_movies)
def get_recommendations_based_on_genres(movie_title , cosine_sim_movies=cosine_sim_movies):
    """
    Calculates top 2 movies to recommend based on given movie titles genres. 
    :param movie_title: title of movie to be taken for base of recommendation
    :param cosine_sim_movies: cosine similarity between movies 
    :return: Titles of movies recommended to user
    """
    global mname
   # movie_title=mname
    print("movie title:"+str(movie_title))
    # Get the index of the movie that matches the title
    idx_movie = df_movies.loc[df_movies['title'].isin([movie_title])]
    idx_movie = idx_movie.index
    #print(idx_movie)
    #print(idx_movie)
    
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores_movies = list(enumerate(cosine_sim_movies[idx_movie][0]))
    
    # Sort the movies based on the similarity scores
    sim_scores_movies = sorted(sim_scores_movies, key=lambda x: x[1], reverse=True)
    #print(sim_scores_movies)

    # Get the scores of the 10 most similar movies
    sim_scores_movies = sim_scores_movies[1:3]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores_movies]
    
    # Return the top 2 most similar movies
    a=df_movies['title'].iloc[movie_indices]
    print(a)

    l5=tk.Label(root,text="Recommended Movie Names:"+ '\n' + str(a),background="green",font=('times', 20, ' bold '),width=0,height=0)
    l5.place(x=400,y=150)
    
    return df_movies['title'].iloc[movie_indices]

def get_recommendations_based_on_genres1(movie_title1 , cosine_sim_movies1=cosine_sim_movies1):
    """
    Calculates top 2 movies to recommend based on given movie titles genres. 
    :param movie_title: title of movie to be taken for base of recommendation
    :param cosine_sim_movies: cosine similarity between movies 
    :return: Titles of movies recommended to user
    """
    global mname
   # movie_title=mname
    print("movie title:"+str(movie_title1))
    # Get the index of the movie that matches the title
    idx_movie1 = df_movies.loc[df_movies['title'].isin([movie_title1])]
    idx_movie1 = idx_movie1.index
    print(idx_movie1)
    #print(idx_movie)
    
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores_movies1 = list(enumerate(cosine_sim_movies[idx_movie1][0]))
    
    # Sort the movies based on the similarity scores
    sim_scores_movies1 = sorted(sim_scores_movies1, key=lambda x: x[1], reverse=True)
    #print(sim_scores_movies)

    # Get the scores of the 10 most similar movies
    sim_scores_movies1 = sim_scores_movies1[1:3]
    
    # Get the movie indices
    movie_indices1 = [i[0] for i in sim_scores_movies1]
    
    # Return the top 2 most similar movies
    a=df_movies['title'].iloc[movie_indices1]
    print(a)

    l5=tk.Label(root,text="Recommended Movie Names:"+'\n'+ str(a),background="purple",font=('times', 20, ' bold '),width=0, height=0)
    l5.place(x=300,y=300)
    
    return df_movies['title'].iloc[movie_indices1]




#get_recommendations_based_on_genres("Father of the Bride Part II (1995)")
#print(get_recommendations_based_on_genres)
#print(df_movies)

def get_recommendation_content_model(userId):
    """
    Calculates top movies to be recommended to user based on movie user has watched.  
    :param userId: userid of user
    :return: Titles of movies recommended to user
    """
    global uid
    recommended_movie_list = []
    movie_list = []
    userId = int(uid)
    print("User Id==========="+str(type(userId)))
    df_rating_filtered = df_ratings[df_ratings["userId"]== userId]
    print("Rating============>"+str(df_rating_filtered))
    for key, row in df_rating_filtered.iterrows():
        movie_list.append((df_movies["title"][row["movieId"]==df_movies["movieId"]]).values) 
    for index, movie in enumerate(movie_list):
        for key, movie_recommended in get_recommendations_based_on_genres1(movie[0]).iteritems():
            recommended_movie_list.append(movie_recommended)

    # removing already watched movie from recommended list    
    for movie_title in recommended_movie_list:
        if movie_title in movie_list:
            recommended_movie_list.remove(movie_title)
    
    return set(recommended_movie_list)

   
#get_recommendation_content_model(1)
# def get_recommendation_content_model(userId):
#     """
#     Calculates top movies to be recommended to user based on movie user has watched.  
#     :param userId: userid of user
#     :return: Titles of movies recommended to user
#     """
#     ratings = pd.read_csv('C:/movie-recommender-system-master/ratings.csv', sep=',', encoding='latin-1', usecols=['userId','movieId','rating','timestamp'])

# # Reading movies file
#     movies = pd.read_csv('C:/movie-recommender-system-master/movies.csv', sep=',', encoding='latin-1', usecols=['movieId','title','genres'])
#     df_movies = movies 
#     df_ratings = ratings 
#     mname=tk.IntVar()
#     #userId=uid.get()
#     print(userId)
#     recommended_movie_list = []
#     movie_list = []
#     df_rating_filtered = df_ratings[df_ratings["userId"]== userId]
#     print("df rating:"+str(df_rating_filtered))
#     
#     for key, row in df_rating_filtered.iterrows():
#         movie_list.append((df_movies["title"][row["movieId"]==df_movies["movieId"]]).values) 
    
#     for index, movie in enumerate(movie_list):
#         for key, movie_recommended in get_recommendations_based_on_genres(movie[0]).iteritems():
#             recommended_movie_list.append(movie_recommended)
#             print("Hello:"+str(movie_recommended))

#     # removing already watched movie from recommended list    
#     for movie_title in recommended_movie_list:
#         if movie_title in movie_list:
#             recommended_movie_list.remove(movie_title)
    
#     return set(recommended_movie_list)
    
#     print(recommended_movie_list)
# get_recommendation_content_model(1)

def movie():
    
    global mname
    mname=mname.get()
    print(mname)
    mname = str(mname)
    print("Movie Name:"+mname)
    get_recommendations_based_on_genres(mname)



def userid():
    
    global uid
    uid=uid.get()
    print(uid)
    uid = int(uid)
    get_recommendation_content_model(uid)



l4=tk.Label(root,text="Userid",background="black",font=('times', 20, ' bold '),fg="white",width=0)
l4.place(x=10,y=100)
uid=tk.Entry(root,bd=2,width=10,font=("TkDefaultFont", 20),textvar=uid)
uid.place(x=100,y=100)

l4=tk.Label(root,text="Movie Name",background="black",font=('times', 20, ' bold '),width=0,fg="white")
l4.place(x=10,y=200)
mname=tk.Entry(root,bd=2,width=15,font=("TkDefaultFont", 20),textvar=mname)
mname.place(x=200,y=200)

button1 = tk.Button(root, text=" Submit ", command=movie,width=15, height=1, font=('times', 15, ' bold '),bg="maroon",fg="white")
button1.place(x=50, y=300)

# button1 = tk.Button(root, text=" Content model ", command=userid,width=15, height=1, font=('times', 15, ' bold '),bg="maroon",fg="white")
# button1.place(x=100, y=550)
    
#Implementation of Item-Item Filtering
df_movies_ratings=pd.merge(df_movies, df_ratings)
a = df_movies_ratings

print(a)


ratings_matrix_items = df_movies_ratings.pivot_table(index=['movieId'],columns=['userId'],values='rating').reset_index(drop=True)
ratings_matrix_items.fillna( 0, inplace = True )
ratings_matrix_items.shape

ratings_matrix_items

# movie_similarity = 1 - pairwise_distances( ratings_matrix_items.as_matrix(), metric="cosine" )
# np.fill_diagonal( movie_similarity, 0 ) #Filling diagonals with 0s for future use when sorting is done
# ratings_matrix_items = pd.DataFrame( movie_similarity )
# ratings_matrix_items


def item_similarity(movieName): 
    """
    recomendates similar movies
   :param data: name of the movie 
   """
    try:
        #user_inp=input('Enter the reference movie title based on which recommendations are to be made: ')
        user_inp=movieName
        inp=df_movies[df_movies['title']==user_inp].index.tolist()
        inp=inp[0]

        df_movies['similarity'] = ratings_matrix_items.iloc[inp]
        df_movies.columns = ['movie_id', 'title', 'release_date','similarity']
    except:
        print("Sorry, the movie is not in the database!")  
def recommendedMoviesAsperItemSimilarity(user_id):
    """
     Recommending movie which user hasn't watched as per Item Similarity
    :param user_id: user_id to whom movie needs to be recommended
    :return: movieIds to user 
    """
    user_movie= df_movies_ratings[(df_movies_ratings.userId==user_id) & df_movies_ratings.rating.isin([5,4.5])][['title']]
    user_movie=user_movie.iloc[0,0]
    item_similarity(user_movie)
    sorted_movies_as_per_userChoice=df_movies.sort_values( ["similarity"], ascending = False )
    sorted_movies_as_per_userChoice=sorted_movies_as_per_userChoice[sorted_movies_as_per_userChoice['similarity'] >=0.45]['movie_id']
    recommended_movies=list()
    df_recommended_item=pd.DataFrame()
    user2Movies= df_ratings[df_ratings['userId']== user_id]['movieId']
    for movieId in sorted_movies_as_per_userChoice:
            if movieId not in user2Movies:
                df_new= df_ratings[(df_ratings.movieId==movieId)]
                df_recommended_item=pd.concat([df_recommended_item,df_new])
            best10=df_recommended_item.sort_values(["rating"], ascending = False )[1:10] 
    return best10['movieId']
def movieIdToTitle(listMovieIDs):
    """
     Converting movieId to titles
    :param user_id: List of movies
    :return: movie titles
    """
    movie_titles=[]
    for id in listMovieIDs:
        movie_titles.append(df_movies[df_movies['movie_id']==id]['title'])
    return movie_titles
def name():
    global user_id
    user_id=user_id.get()
    a =movieIdToTitle(recommendedMoviesAsperItemSimilarity(user_id))
    df = pd.DataFrame(a)
    print(type(a))
    print(df)
    


 
    l4=tk.Label(root,text=a,background="black",font=('times', 20, ' bold '),width=0,fg="white")
    l4.place(x=0,y=100)
    print("Recommended movies,:\n",a)

l4=tk.Label(root,text="Userid",background="purple",font=('times', 20, ' bold '),width=0)
l4.place(x=10,y=450)
uid1=tk.Entry(root,bd=2,width=35,font=("TkDefaultFont", 20),textvar=user_id)
uid1.place(x=100,y=450)
button1 = tk.Button(root, text=" Submit ", command=name,width=15, height=1, font=('times', 15, ' bold '),bg="black",fg="white")
button1.place(x=100, y=600)
#login=tk.Button(root,text="Login",width=30,height=5,bg='light blue',fg="black",command = get_recommendation_content_model).place(x=180,y=350)











#################Implementation of User-Item Filtering#############################
ratings_matrix_users = df_movies_ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating').reset_index(drop=True)
ratings_matrix_users.fillna( 0, inplace = True )
# movie_similarity = 1 - pairwise_distances( ratings_matrix_users.as_matrix(), metric="cosine" )
# np.fill_diagonal( movie_similarity, 0 ) #Filling diagonals with 0s for future use when sorting is done
# ratings_matrix_users = pd.DataFrame( movie_similarity )
# ratings_matrix_users
ratings_matrix_users.idxmax(axis=1)
ratings_matrix_users.idxmax(axis=1).sample( 10, random_state = 10 )
similar_user_series= ratings_matrix_users.idxmax(axis=1)
df_similar_user= similar_user_series.to_frame()
df_similar_user.columns=['similarUser']
df_similar_user









#login=tk.Button(root,text="Login",width=30,height=5,bg='light blue',fg="black",command = get_recommendation_content_model).place(x=180,y=350)

root.mainloop()

