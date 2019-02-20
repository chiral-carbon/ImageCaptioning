from tkinter import *
from tkinter import ttk
from tkinter import filedialog
gui = Tk()
gui.geometry("370x50")
gui.title("Find the file/directory")

def getFolderPath():
    folder_selected = filedialog.askdirectory()
    folderPath.set(folder_selected)

def doStuff():
    folder = folderPath.get()
    print("Selected: ", folder)

print("Browse the desired file or directory with the 'Browse' button.\nClick on 'Find' to use the browsed file/directory in your program.")

folderPath = StringVar()

a = Label(gui ,text="Enter name: ")
a.grid(row=0,column = 0)
E = Entry(gui,textvariable=folderPath)
E.grid(row=0,column=1)
btnFind = ttk.Button(gui, text="Browse",command=getFolderPath)
btnFind.grid(row=0,column=2)
c = ttk.Button(gui ,text="Find", command=doStuff)
c.grid(row=4,column=0)

gui.mainloop()