
import pandas as pd
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tkinter as tk
from PIL import Image, ImageTk

from pandas import *
import pandas as pd
import csv
#import gmplot
# Open the csv file
import os
#from tkvideo import tkvideo


root = tk.Tk()
#root.geometry("1400x500")
root.title("Movie Recommandation System")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Home Page")

   
z1 = tk.StringVar()
#uid=tk.IntVar()
#user_id=tk.IntVar()


# #####For background Image
image2 = Image.open('slide1.jpg')
image2 = image2.resize((w,h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
# video_label =tk.Label(root)
# video_label.pack()
# # read video to display on label
# player = tkvideo("D:/Movie_recommandation/Movie_recommandation/mov_vi.gif", video_label,loop = 1, size = (w, h))
# player.play()

#
label_l2 = tk.Label(root, text="___Movie Recommendation System___",font=("times", 30, 'bold'),
                    background="#FF4500", fg="black", width=30, height=2)
label_l2.place(x=450, y=0)


    

def KNN():
    from sklearn.neighbors import KNeighborsClassifier  
    def get_movie_label(movie_id):
        """
        Get the cluster label to which movie belongs by KNN algorithm.  
        :param movie_id: movie id
        :return: genres label to movie belong
        """
        classifier = KNeighborsClassifier(n_neighbors=5)
        x= tfidf_movies_genres_matrix
        y = df_movies.iloc[:,-1]
        classifier.fit(x, y)
        y_pred = classifier.predict(tfidf_movies_genres_matrix[movie_id])
        return y_pred
    true_count = 0
    false_count = 0
    def evaluate_based_model():
        """
        Evaluate content based model.  
        """
        for key, colums in df_movies.iterrows():
            movies_recommended_by_model = get_recommendations_based_on_genres(colums["title"])
            predicted_genres  = get_movie_label(movies_recommended_by_model.index)
            for predicted_genre in predicted_genres:
                global true_count, false_count
                if predicted_genre == colums["genres"]:
                    true_count = true_count+1
                else:
    #                 print(colums["genres"])
    #                 print(predicted_genre)
                    false_count = false_count +1
    evaluate_content_based_model()
    total = true_count + false_count
   
    



def recomm():
    z = z1.get()
    movie_csv = pd.read_csv(r'movies.csv')
   # z ="Baby-Sitters Club, The (1995)"
    title =movie_csv[movie_csv["title"] == z]
    title = title.drop(['title'], axis = 1)
    print(title)
    for movie_id in title['movieId'].values:
        print(movie_id)
    data = pd.read_csv(r'Book1.csv')
    interestingRow = data[data["movieId"] == movie_id]
    print (interestingRow)
    df = pd.DataFrame(interestingRow)
    #df = df.head(3)
    df = df.iloc[[1, 2]]
    print(df)
    df = df.drop(['movieId'], axis = 1)
    print(df)
    for z in df['userId'].values:
        recommandated =data[data["userId"] == z]
        print(recommandated)
    final = recommandated.drop(['userId'], axis = 1)
    print("=========================Final Recommanded============================================")
    final3 = final.sample(n=10)
    final3.to_csv('recommanded_movie.csv', sep=',', encoding='utf-8', index=False)
    print(final3)
    m1 = pd.read_csv(r'movies.csv')
    os.remove('D:/Movie_recommandation/mov.txt')
    for y in final3['movieId'].values:
        recommandated_movie_name =m1[m1["movieId"] == y]
        recom = recommandated_movie_name.drop(['movieId'], axis = 1)
        
        for m in recom['title'].values:
            a = m
            print(a)
            file = open(r"D:/Movie_recommandation/mov.txt", 'a+')
            file.write(m + '\n')
            file.close()
            
    file1 = open(r"D:/Movie_recommandation/mov.txt", 'r', encoding='UTF8').read()
    file.close() 

   # file1.read(m + '\n')
    #print(file1)
    l4=tk.Label(root,text="============== Recommended Movies ============="+'\n\n'+str(file1),background="#B0E0E6",font=('times', 20, ' bold '),width=0,fg="black").grid(row=0,padx=700,pady=300)
    #l4.place(x=0,y=500)

l4=tk.Label(root,text="Movie Name",background="white",fg="black",font=('times', 20, ' bold '),width=0)
l4.place(x=100,y=200)
uid1=tk.Entry(root,bd=2,width=35,font=("TkDefaultFont", 20),textvar=z1)
uid1.place(x=300,y=200)
button1 = tk.Button(root, text=" Submit ", command=recomm,width=15, height=1, font=('times', 15, ' bold '),bg="#8B2252",fg="white")
button1.place(x=350, y=300)
root.mainloop()
