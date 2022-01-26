import tkinter as tk
from os.path import isfile
from tkinter import Button, Text, LEFT, RIGHT, TOP, filedialog, NORMAL, END, INSERT, DISABLED, messagebox

import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from sklearn.preprocessing import StandardScaler, LabelEncoder

from preprocessing import preprocessing, train_model


def styling_widgets(wid):
    wid['bg'] = "#E8E8E8"
    for widget in wid.winfo_children():
        name = type(widget).__name__
        if widget.winfo_children():
            styling_widgets(widget)
        if name == "Button":
            widget['bg'] = "#E4E4E4"

        elif name == "Text":
            widget['bg'] = "#FFF"
        else:
            widget['bg'] = "#E8E8E8"

        if hasattr(widget, "font"):
            widget['font'] = 'Roboto', 10


def handle_dialogs(file_selected, text_widget):
    text_widget.config(state=NORMAL)
    text_widget.delete("1.0", END)
    text_widget.insert(INSERT, file_selected)
    text_widget.config(state=DISABLED)


class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.width, self.height = 900, 600
        self.geometry("%dx%d" % (self.width, self.height))
        self.resizable(False, False)
        self.title("NBA All Star Prediction")
        self.df = None

        img_name = "./src/title.png"
        load_img = Image.open(img_name)
        self.render = ImageTk.PhotoImage(load_img)

        self.pane1 = tk.PanedWindow(self)
        self.pane2 = tk.PanedWindow(self)
        self.get_dataset_btn = Button(self.pane1, text="Add Dataset", width=30, height=3, command=self.open_file_dialog)
        self.get_dataset_txt = Text(self.pane1, width=60, height=3)
        self.predict_btn = Button(self.pane2, text="Make Prediction", width=30, height=3, command=self.predict_data)

        self.place_widgets()
        styling_widgets(self)

    def place_widgets(self):
        self.get_dataset_btn.pack(side=LEFT)
        self.get_dataset_txt.pack(side=RIGHT)

        self.predict_btn.pack(side=RIGHT)

        img = tk.Label(self, image=self.render, bg="#E8E8E8")
        img.image = self.render
        img.pack(side="top")

        self.pane1.pack(side=TOP, pady=45)
        self.pane2.pack(side=TOP, pady=20)

        self.get_dataset_txt.config(state=DISABLED)

    def open_file_dialog(self):
        root = tk.Tk()
        root.withdraw()
        try:
            file_selected = filedialog.askopenfilename(
                title="Select Dataset",
                filetypes=(("CSV files", "*.csv"), ("All Files", "*.*"))
            )
            if isfile(file_selected):
                handle_dialogs(file_selected, self.get_dataset_txt)
            else:
                handle_dialogs("", self.get_dataset_txt)

        except FileNotFoundError:
            handle_dialogs("", self.get_dataset_txt)
        except TypeError:
            messagebox.showerror("Not a valid file",
                                 "Make sure you're selecting a valid file")

    def predict_data(self):
        path = self.get_dataset_txt.get("1.0", END).strip()
        main_df = pd.read_csv(path)
        main_df = preprocessing(main_df)
        df = main_df.copy()
        label_encoder = LabelEncoder()

        df['Tm'] = label_encoder.fit_transform(df['Tm'])
        df['Pos'] = label_encoder.fit_transform(df['Pos'])
        df['Player'] = label_encoder.fit_transform(df['Player'])

        aha = df.drop(columns=['Player', 'Age', 'Year'])
        model = train_model()
        sc = StandardScaler()
        aha = sc.fit_transform(aha)
        vals = model.predict(np.array(aha))

        main_df['Prediction'] = vals
        names = sorted(list(main_df[main_df['Prediction'] == 1.0]['Player']))

        messagebox.showinfo("All Stars Selected", '\n'.join(map(str, names)))


if __name__ == '__main__':
    app = App()
    app.mainloop()
