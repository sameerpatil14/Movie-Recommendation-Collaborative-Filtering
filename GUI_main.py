import tkinter as tk
#from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk



##############################################+=============================================================
root = tk.Tk()
root.configure(background="black")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Home Page")

# 43

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
# image2 = Image.open('crop5.jpg')
# image2 = image2.resize((w,h), Image.ANTIALIAS)

# background_image = ImageTk.PhotoImage(image2)

# background_label = tk.Label(root, image=background_image)

# background_label.image = background_image

# background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)



#
label_l2 = tk.Label(root, text="Movie recommendation system Using Collaborative Filtering",font=("times", 30, 'bold'),
                    background="#8B0A50", fg="white", width=70, height=2)
label_l2.place(x=0, y=0)

img = Image.open('logo.jpg')
img = img.resize((100,75), Image.ANTIALIAS)
logo_image = ImageTk.PhotoImage(img)

logo_label = tk.Label(root, image=logo_image)
logo_label.image = logo_image
logo_label.place(x=21, y=10)




img=ImageTk.PhotoImage(Image.open("slide1.jpg"))

img2=ImageTk.PhotoImage(Image.open("slide2.jpg"))

img3=ImageTk.PhotoImage(Image.open("slide3.jpg"))


logo_label=tk.Label()
logo_label.place(x=0,y=95)




# using recursion to slide to next image
x = 1

# function to change to next image
def move():
	global x
	if x == 4:
		x = 1
	if x == 1:
		logo_label.config(image=img,width=1800,height=700)
	elif x == 2:
		logo_label.config(image=img2,width=1800,height=700)
	elif x == 3:
		logo_label.config(image=img3,width=1800,height=700)
	x = x+1
	root.after(2000, move)

# calling the function
move()

frame_alpr = tk.LabelFrame(root, text=" --Login & Register-- ", width=800, height=300, bd=5, font=('times', 14, ' bold '),bg="pink")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=400, y=400)


#T1.tag_configure("center", justify='center')
#T1.tag_add("center", 1.0, "end")

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def log():
    from subprocess import call
    call(["python","login.py"])
    #root.destroy()
    
def reg():
    from subprocess import call
    call(["python","registration.py"])
    #root.destroy()
    
def window():
  root.destroy()
  
  

#####################################################################################################################

button1 = tk.Button(frame_alpr, text="Login", command=log, width=15, height=1,font=('times', 15, ' bold '), bg="black", fg="white")
button1.place(x=100, y=20)

button2 = tk.Button(frame_alpr, text="Registration",command=reg,width=15, height=1,font=('times', 15, ' bold '), bg="black", fg="white")
button2.place(x=300, y=100)

button3 = tk.Button(frame_alpr, text="Exit",command=window,width=14, height=1,font=('times', 15, ' bold '), bg="red", fg="white")
button3.place(x=450, y=200)


label_l1 = tk.Label(root, text="** Movie recommendation system Using Collaborative Filtering **",font=("Times New Roman", 10, 'bold'),
                    background="black", fg="white", width=100, height=2)
label_l1.place(x=400, y=798)


root.mainloop()