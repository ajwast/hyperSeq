import tkinter as tk

class SequencerGUI:
    def __init__(self, root, rhythm, label="Sequencer"):
        self.frame = tk.Frame(root)
        self.frame.pack(pady=10)

        self.label = tk.Label(self.frame, text=label, font=("Arial", 14))
        self.label.pack()

        # Rhythm boxes
        self.canvas = tk.Canvas(self.frame, width=400, height=40, bg="black")
        self.canvas.pack(pady=5)
        self.boxes = []
        box_width = 400 // len(rhythm)
        for i, step in enumerate(rhythm):
            x1, y1 = i * box_width, 0
            x2, y2 = x1 + box_width, 40
            color = "white" if step == 1 else "gray"
            box = self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
            self.boxes.append(box)

        # Buttons
        self.btn_frame = tk.Frame(self.frame)
        self.btn_frame.pack(pady=5)

        self.pitch_btn = tk.Button(self.btn_frame, text="Generate Pitches")
        self.pitch_btn.grid(row=0, column=0, padx=5)

        self.rhythm_btn = tk.Button(self.btn_frame, text="Generate Rhythm")
        self.rhythm_btn.grid(row=0, column=1, padx=5)

    def update(self, step, rhythm):
        for i, (box, val) in enumerate(zip(self.boxes, rhythm)):
            color = "white" if val == 1 else "gray"
            if i == step:
                color = "green"
            self.canvas.itemconfig(box, fill=color)
