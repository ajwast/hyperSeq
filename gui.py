import tkinter as tk

class SequencerGUI:
    def __init__(self, root, rhythm, label="Seq", cell_size=30):
        self.root = root
        self.rhythm = rhythm
        self.cell_size = cell_size

        frame = tk.Frame(root)
        frame.pack()

        self.label = tk.Label(frame, text=label, fg="white", bg="black")
        self.label.pack()

        self.canvas = tk.Canvas(
            frame, width=16 * cell_size, height=cell_size, bg="black"
        )
        self.canvas.pack()

        self.cells = []
        for i in range(16):
            x0, y0 = i * cell_size, 0
            x1, y1 = x0 + cell_size, cell_size
            fill = "green" if rhythm[i] == 1 else "gray"
            rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="white")
            self.cells.append(rect)

        self.highlight = None

    def update(self, step, rhythm=None):
        if rhythm is not None:
            self.rhythm = rhythm
            for i, r in enumerate(rhythm):
                fill = "green" if r == 1 else "gray"
                self.canvas.itemconfig(self.cells[i], fill=fill)

        if self.highlight is not None:
            self.canvas.itemconfig(self.cells[self.highlight], width=1)

        self.canvas.itemconfig(self.cells[step], width=3)
        self.highlight = step
