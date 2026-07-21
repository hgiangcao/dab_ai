import tkinter as tk
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class DotsAndBoxesGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dots & Boxes Agent Interface")
        
        self.driver = webdriver.Chrome()
        self.driver.get("https://www.math.ucla.edu/~tom/Games/dots&boxes.html")
        
        wait = WebDriverWait(self.driver, 10)
        
        alert = wait.until(EC.alert_is_present())
        alert.send_keys("5")
        alert.accept()
        
        alert = wait.until(EC.alert_is_present())
        alert.send_keys("5")
        alert.accept()

        self.canvas = tk.Canvas(root, width=320, height=320, bg="white")
        self.canvas.pack(pady=10)
        
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)
        
        tk.Label(control_frame, text="Move ID:").pack(side=tk.LEFT)
        self.move_entry = tk.Entry(control_frame, width=10)
        self.move_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Submit", command=self.submit_move).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Random", command=self.random_move).pack(side=tk.LEFT, padx=5)
        tk.Button(root, text="Sync State", command=self.redraw_state).pack(pady=5)

        self.waiting_for_opponent = False
        self.valid_moves = []
        self.redraw_state()

    def redraw_state(self):
        self.canvas.delete("all")
        offset = 40
        spacing = 45
        
        self.drawn_lines = {}
        self.valid_moves = []
        
        images = self.driver.find_elements(By.TAG_NAME, "img")
        for img in images:
            name = img.get_attribute("name")
            src = img.get_attribute("src")
            if name:
                if "red" in src:
                    self.drawn_lines[name] = "red"
                elif "blue" in src:
                    self.drawn_lines[name] = "blue"
                elif "black" in src:
                    self.drawn_lines[name] = "black"
                elif "blank.gif" in src:
                    self.valid_moves.append(name)

        # Draw Grid Lines
        for r in range(6):
            for c in range(6):
                x = offset + c * spacing
                y = offset + r * spacing
                
                if c < 5:
                    h_id = f"he{r}{c}"
                    color = self.drawn_lines.get(h_id, "#E0E0E0")
                    width = 3 if h_id in self.drawn_lines else 1
                    self.canvas.create_line(x, y, x + spacing, y, fill=color, width=width)

                if r < 5:
                    v_id = f"ve{r}{c}"
                    color = self.drawn_lines.get(v_id, "#E0E0E0")
                    width = 3 if v_id in self.drawn_lines else 1
                    self.canvas.create_line(x, y, x, y + spacing, fill=color, width=width)

        # Draw Dots
        for r in range(6):
            for c in range(6):
                x = offset + c * spacing
                y = offset + r * spacing
                self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="black")

        # Unlock interface
        self.waiting_for_opponent = False
        self.move_entry.config(state=tk.NORMAL)

    def submit_move(self, move_id=None):
        if self.waiting_for_opponent:
            return

        if move_id is None:
            move_id = self.move_entry.get().strip().lower()
            if len(move_id) == 3 and move_id[0] in ['h', 'v']:
                move_id = move_id[0] + 'e' + move_id[1:]

        if not move_id or move_id not in self.valid_moves:
            return

        try:
            # Lock interface while waiting for computer
            self.waiting_for_opponent = True
            self.move_entry.config(state=tk.DISABLED)
            
            wait = WebDriverWait(self.driver, 2)
            element = wait.until(EC.element_to_be_clickable((By.NAME, move_id)))
            element.click()
        except Exception:
            print(f"Error: Move '{move_id}' failed.")
            self.waiting_for_opponent = False
            self.move_entry.config(state=tk.NORMAL)
            
        self.move_entry.delete(0, tk.END)
        # Wait 1000ms for computer to finish its move before redrawing
        self.root.after(1000, self.redraw_state)

    def random_move(self):
        if self.valid_moves and not self.waiting_for_opponent:
            move = random.choice(self.valid_moves)
            self.submit_move(move)

    def on_close(self):
        self.driver.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DotsAndBoxesGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()