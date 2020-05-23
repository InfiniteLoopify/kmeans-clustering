import tkinter as tk
from tkinter import ttk
from kmean import *


class Table:
    def __init__(self):
        self.scores = tk.Tk()
        self.scores.resizable(False, False)
        self.cols = ('No.', 'Document Name',
                     'Document Title', 'Expected', 'Predicted', '')
        self.listBox = ttk.Treeview(
            self.scores, columns=self.cols, show='headings', height=22)

    def display_Table(self, result, param, label, label_id, classes_names):
        classes = {i: classes_names[i] for i in range(0, len(classes_names))}
        # if not result:
        #     self.listBox.insert("", "end", values=(
        #         '-', '-', '-', '-', '-'))
        for i, ans in enumerate(result):
            left_str = str(classes[int(label[i])])
            right_str = "0" * \
                (3-len(str(int(label_id[i])))) + str(int(label_id[i]))+".txt"
            path = left_str+"\t>  "+right_str
            file_name = left_str+"/"+right_str
            f = open('bbcsport/'+file_name, "r")
            title = f.readline()
            correct = "\u25cf"
            tg = 'false'
            if int(ans) == int(label[i]):
                correct = "\u2714"
                tg = 'true'
            self.listBox.insert("", "end", tags=(tg,), values=(
                i+1, path, title, classes[int(label[i])], classes[int(ans)], correct))
        self.listBox.tag_configure('false', background='#FFEFEE')
        self.listBox.tag_configure('true', background='#F6FFF4')

    def create_Gui(self, indexer):
        self.scores.geometry("1020x600")
        self.scores.title('Vector Space Model')

        self.label = tk.Label(self.scores, text="K-Mean", font=(
            "Arial", 30)).grid(row=0, columnspan=2)

        # answer = tk.StringVar()
        # searchQuery = tk.Entry(
        #     self.scores, width=94, textvariable=answer).place(x=10, y=52)

        self.label = tk.Label(self.scores, text="").grid(
            row=1, column=2, pady=5)

        # self.label = tk.Label(self.scores, text="KNN Value:\t"+"3",font=("Arial", 10)).place(x=694+160, y=48-40)
        # self.label = tk.Label(self.scores, text="Training Data:\t" + str(int(indexer.param[0]*100))+"%",font=("Arial", 10)).place(x=694+160, y=48-20)
        # self.label = tk.Label(self.scores, text="Testing Data:\t" + str(int(100-indexer.param[0]*100))+"%",font=("Arial", 10)).place(x=694+160, y=48-0)
        self.label = tk.Label(self.scores, text=f"{indexer.param[3]:0.2f}%", fg="#275A2C",
                              font=("Helvetica", 30)).place(x=694+160, y=545)

        # answer2 = tk.StringVar(value="0.0005")
        # searchQuery2 = tk.Entry(
        #     self.scores, width=10, textvariable=answer2, justify='right').place(x=710+205, y=50-35)

        vsb = ttk.Scrollbar(
            self.scores, orient="vertical", command=self.listBox.yview)
        vsb.place(x=1004, y=79, height=460)
        vsb.configure(command=self.listBox.yview)
        self.listBox.configure(yscrollcommand=vsb.set)

        for col in self.cols:
            self.listBox.heading(col, text=col)
        self.listBox.grid(row=2, column=0, columnspan=2)
        self.listBox.column(self.cols[0], minwidth=40, width=60, stretch=tk.NO)
        self.listBox.column(
            self.cols[1], minwidth=120, width=180, stretch=tk.NO)
        self.listBox.column(self.cols[2], minwidth=120, width=522)
        self.listBox.column(self.cols[3], minwidth=75, width=110)
        self.listBox.column(self.cols[4], minwidth=75, width=110)
        self.listBox.column(self.cols[5], minwidth=20, width=20)

        """ showScores = tk.Button(self.scores, text="Search", width=22,
                               command=lambda:
                               [
                                   self.listBox.delete(
                                       *self.listBox.get_children()),
                                   tb.display_Table(
                                       indexer.calculate(answer.get(), float(answer2.get())))
                               ]).place(x=798, y=45) """
        closeButton = tk.Button(self.scores, text="Close", width=15,
                                command=exit).grid(row=4, column=0, columnspan=2)

        self.display_Table(indexer.result, indexer.param,
                           indexer.label, indexer.label_id, indexer.classes_names)
        self.scores.mainloop()


if __name__ == "__main__":
    indexer = Indexer()
    indexer.read_file('bbcsport/', 'files/')

    tb = Table()
    tb.create_Gui(indexer)
