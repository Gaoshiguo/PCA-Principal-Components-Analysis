import tkinter
from tkinter.filedialog import *
from tkinter import *

root = tkinter.Tk()
def file_save():
    f = tkinter.filedialog.asksaveasfile(mode='w', filetypes=[("txt",".txt"),("csv",".csv"),("PNG",".png"),("GPF",".gpf"),("JPG",".jpg"),("python",".py")])
    if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    text2save = str(text.get(1.0, END)) # starts from `1.0`, not `0.0`
    f.write(text2save)
    f.close() # `()` was missing.

label=Label(root,text="请输入您想要保存的文字：")
label.place(relx=0.3,rely=0.01,relwidth=0.3,relheight=0.2)
text = Text(root,width=20,height=15)
text.place(relx=0.3,rely=0.2,relwidth=0.5,relheight=0.5)

button=Button(root,text="选择想要复制的存储路径",command=file_save)
button.place(relx=0.3,rely=0.7)
root.title = ('记事本')
root.geometry('640x480')
root.mainloop()