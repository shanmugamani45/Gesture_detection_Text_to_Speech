# Importing Libraries
import numpy as np
import math
import cv2
import os, sys
import traceback
import pyttsx3
from keras.models import load_model
from HandTrackingModule import HandDetector
from string import ascii_uppercase
import threading
from spellchecker import SpellChecker
class SpellCheckerAdapter:
    def __init__(self):
        self.spell = SpellChecker()

    def check(self, word):
        return word.lower() in self.spell

    def suggest(self, word):
        word_lower = word.lower()
        # For correctly spelled words, find words that start with the same letters
        # For misspelled words, get correction candidates
        candidates = self.spell.candidates(word_lower)
        if candidates:
            results = sorted(candidates, key=lambda w: self.spell.word_frequency[w], reverse=True)
        else:
            results = []

        # Also add completions: words in the dictionary that start with the typed letters
        if len(word_lower) >= 2:
            prefix_matches = [
                w for w in self.spell.word_frequency.keys()
                if w.startswith(word_lower) and w != word_lower
            ]
            # Sort by frequency (most common first)
            prefix_matches.sort(key=lambda w: self.spell.word_frequency[w], reverse=True)
            # Merge: exact word first, then prefix matches, then spell candidates
            merged = []
            if word_lower in self.spell:
                merged.append(word_lower)
            for w in prefix_matches[:6]:
                if w not in merged:
                    merged.append(w)
            for w in results:
                if w not in merged:
                    merged.append(w)
            return [w.upper() for w in merged[:4]]
        return [w.upper() for w in results[:4]]
ddd = SpellCheckerAdapter()
hd = HandDetector(maxHands=1)
import tkinter as tk
from PIL import Image, ImageTk

offset=29

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


# Application :

class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('cnn8grps_rad1_model.h5')

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag=False
        self.next_flag=True
        self.prev_char=""
        self.count=-1
        self.ten_prev_char=[]
        for i in range(10):
            self.ten_prev_char.append(" ")
            
        self.stable_counter = 0
        self.stable_pred = ""

        for i in ascii_uppercase:
            self.ct[i] = 0

        print("Loaded model from disk")

        # ── Colour tokens ──────────────────────────────────────────────────
        BG          = "#0d1117"
        PANEL_BG    = "#161b22"
        ACCENT      = "#58a6ff"
        ACCENT2     = "#3fb950"
        TEXT_PRI    = "#e6edf3"
        TEXT_SEC    = "#8b949e"
        BORDER      = "#30363d"
        BTN_BG      = "#21262d"
        BTN_HOV     = "#30363d"
        DANGER      = "#f85149"
        CHAR_BG     = "#1c2128"

        self.root = tk.Tk()
        self.root.title("ASL Gesture Recognition")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1400x820")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.root.bind('<Return>',    self.commit_char)
        self.root.bind('<space>',     self.add_space)
        self.root.bind('<BackSpace>', self.delete_char)

        # ── Title bar ──────────────────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg=PANEL_BG, height=60)
        title_bar.place(x=0, y=0, width=1400, height=60)
        tk.Frame(self.root, bg=ACCENT, height=2).place(x=0, y=60, width=1400)

        tk.Label(title_bar, text="🤟  ASL Gesture Recognition",
                 font=("Segoe UI", 20, "bold"),
                 bg=PANEL_BG, fg=TEXT_PRI).place(x=20, y=12)

        tk.Label(title_bar, text="Enter = save  •  Space = space  •  Backspace = delete",
                 font=("Segoe UI", 11),
                 bg=PANEL_BG, fg=TEXT_SEC).place(x=560, y=20)

        # ── Left panel: webcam feed ────────────────────────────────────────
        cam_frame = tk.Frame(self.root, bg=BORDER, bd=0)
        cam_frame.place(x=20, y=75, width=504, height=504)
        tk.Frame(cam_frame, bg=PANEL_BG).place(x=2, y=2, width=500, height=500)

        self.panel = tk.Label(cam_frame, bg="#000000")
        self.panel.place(x=2, y=2, width=500, height=500)

        # ── Centre panel: hand skeleton ───────────────────────────────────
        skel_outer = tk.Frame(self.root, bg=BORDER)
        skel_outer.place(x=544, y=75, width=304, height=304)
        self.panel2 = tk.Label(skel_outer, bg=PANEL_BG)
        self.panel2.place(x=2, y=2, width=300, height=300)

        tk.Label(self.root, text="Hand Skeleton",
                 font=("Segoe UI", 10), bg=BG, fg=TEXT_SEC).place(x=665, y=383)

        # ── Detected character display ─────────────────────────────────────
        char_outer = tk.Frame(self.root, bg=ACCENT, bd=0)
        char_outer.place(x=544, y=405, width=304, height=174)

        char_inner = tk.Frame(char_outer, bg=CHAR_BG)
        char_inner.place(x=2, y=2, width=300, height=170)

        tk.Label(char_inner, text="Detected",
                 font=("Segoe UI", 11), bg=CHAR_BG, fg=TEXT_SEC).place(x=0, y=10, width=300)

        self.panel3 = tk.Label(char_inner, text="–",
                               font=("Segoe UI", 72, "bold"),
                               bg=CHAR_BG, fg=ACCENT)
        self.panel3.place(x=0, y=30, width=300, height=120)

        # ── Right panel: sentence + suggestions ────────────────────────────
        right_x = 868

        # Sentence card
        sent_card = tk.Frame(self.root, bg=BORDER)
        sent_card.place(x=right_x, y=75, width=512, height=160)
        tk.Frame(sent_card, bg=PANEL_BG).place(x=2, y=2, width=508, height=156)

        tk.Label(sent_card, text="Sentence",
                 font=("Segoe UI", 11, "bold"), bg=PANEL_BG, fg=ACCENT).place(x=12, y=10)

        self.panel5 = tk.Label(sent_card, text="",
                               font=("Segoe UI", 22, "bold"),
                               bg=PANEL_BG, fg=TEXT_PRI,
                               anchor="w", wraplength=480, justify="left")
        self.panel5.place(x=10, y=38, width=488, height=108)

        # Action buttons row
        def make_action_btn(parent, text, cmd, color=ACCENT):
            f = tk.Frame(parent, bg=color)
            btn = tk.Button(f, text=text,
                            font=("Segoe UI", 12, "bold"),
                            bg=BTN_BG, fg=color,
                            relief="flat", bd=0, cursor="hand2",
                            activebackground=BTN_HOV, activeforeground=color,
                            command=cmd)
            btn.place(x=1, y=1, width=116, height=36)
            return f

        act_row = tk.Frame(self.root, bg=BG)
        act_row.place(x=right_x, y=245, width=512, height=40)

        self.speak_btn_frame = make_action_btn(act_row, "🔊  Speak",    self.speak_fun, ACCENT2)
        self.speak_btn_frame.place(x=0,   y=0, width=118, height=38)
        self.clear_btn_frame = make_action_btn(act_row, "✕  Clear",     self.clear_fun, DANGER)
        self.clear_btn_frame.place(x=126, y=0, width=118, height=38)

        # Suggestions card
        sug_card = tk.Frame(self.root, bg=BORDER)
        sug_card.place(x=right_x, y=300, width=512, height=280)
        tk.Frame(sug_card, bg=PANEL_BG).place(x=2, y=2, width=508, height=276)

        tk.Label(sug_card, text="Suggestions",
                 font=("Segoe UI", 11, "bold"), bg=PANEL_BG, fg=ACCENT).place(x=12, y=10)

        def make_sug_btn(parent, x, y, w, h):
            f = tk.Frame(parent, bg=BORDER)
            f.place(x=x, y=y, width=w, height=h)
            inner = tk.Frame(f, bg=BTN_BG)
            inner.place(x=1, y=1, width=w-2, height=h-2)
            btn = tk.Button(inner, text="",
                            font=("Segoe UI", 14, "bold"),
                            bg=BTN_BG, fg=TEXT_PRI,
                            relief="flat", bd=0, cursor="hand2",
                            activebackground=BTN_HOV, activeforeground=ACCENT)
            btn.place(x=0, y=0, width=w-2, height=h-2)
            return btn

        self.b1 = make_sug_btn(sug_card, 10,  36, 228, 108)
        self.b2 = make_sug_btn(sug_card, 266, 36, 228, 108)
        self.b3 = make_sug_btn(sug_card, 10,  154, 228, 108)
        self.b4 = make_sug_btn(sug_card, 266, 154, 228, 108)

        # ── Webcam bottom strip: character label ───────────────────────────
        strip = tk.Frame(self.root, bg=PANEL_BG, height=40)
        strip.place(x=20, y=579, width=504, height=40)

        tk.Label(strip, text="Detecting:",
                 font=("Segoe UI", 12), bg=PANEL_BG, fg=TEXT_SEC).place(x=10, y=10)

        self.lbl_char_strip = tk.Label(strip, text="–",
                                       font=("Segoe UI", 14, "bold"),
                                       bg=PANEL_BG, fg=ACCENT)
        self.lbl_char_strip.place(x=110, y=8)

        # ── Bottom keyboard hint ───────────────────────────────────────────
        hint = tk.Frame(self.root, bg=PANEL_BG, height=36)
        hint.place(x=0, y=784, width=1400, height=36)
        tk.Frame(self.root, bg=BORDER, height=1).place(x=0, y=783, width=1400)

        for col, (key, desc) in enumerate([("Enter","Commit letter"),("Space","Add space"),("Backspace","Delete")]):
            x_pos = 40 + col * 230
            kf = tk.Frame(hint, bg=BORDER)
            kf.place(x=x_pos, y=6, width=80, height=24)
            tk.Frame(kf, bg=BTN_BG).place(x=1, y=1, width=78, height=22)
            tk.Label(kf, text=key, font=("Segoe UI", 9, "bold"), bg=BTN_BG, fg=ACCENT).place(x=1,y=1,width=78,height=22)
            tk.Label(hint, text=desc, font=("Segoe UI", 9), bg=PANEL_BG, fg=TEXT_SEC).place(x=x_pos+88, y=11)

        # ── State vars ─────────────────────────────────────────────────────
        self.str = " "
        self.ccc = 0
        self.word = " "
        self.current_symbol = "–"
        self.photo = "Empty"

        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            cv2image = cv2.flip(frame, 1)
            hands = hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy=np.array(cv2image)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands:
                # #print(" --------- lmlist=",hands[1])
                hand = hands[0]
                x, y, w, h = hand['bbox']
                white = cv2.imread("white.jpg")

                print(" ", self.ccc)
                self.ccc += 1
                
                self.pts = []
                for point in hand['lmList']:
                    cropped_x = point[0] - (x - offset)
                    cropped_y = point[1] - (y - offset)
                    self.pts.append([cropped_x, cropped_y, point[2]])

                os = ((400 - w) // 2) - 15
                os1 = ((400 - h) // 2) - 15
                if True:
                    for t in range(0, 4, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1), (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1), (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1),
                             (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0),
                             3)

                    for i in range(21):
                        cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 0, 255), 1)

                    res=white
                    self.predict(res)

                    self.current_image2 = Image.fromarray(res)

                    imgtk = ImageTk.PhotoImage(image=self.current_image2)

                    self.panel2.imgtk = imgtk
                    self.panel2.config(image=imgtk)

                    self.panel3.config(text=self.current_symbol)
                    self.lbl_char_strip.config(text=self.current_symbol)

                    self.b1.config(text=self.word1, command=self.action1)
                    self.b2.config(text=self.word2, command=self.action2)
                    self.b3.config(text=self.word3, command=self.action3)
                    self.b4.config(text=self.word4, command=self.action4)
        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.panel5.config(text=self.str.strip())
            self.root.after(1, self.video_loop)

    def distance(self,x,y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def commit_char(self, event):
        if self.current_symbol and self.current_symbol not in [" ", "next", "Backspace"]:
            self.str += self.current_symbol

    def add_space(self, event):
        self.str += " "

    def delete_char(self, event):
        if len(self.str) > 1:
            self.str = self.str[:-1]
        elif len(self.str) == 1:
            self.str = " "

    def action1(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word1.upper()


    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str=self.str[:idx_word]
        self.str=self.str+self.word2.upper()
        #self.str[idx_word:last_idx] = self.word2


    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word3.upper()



    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word4.upper()


    def speak_fun(self):
        text = self.str.strip()
        if not text:
            return
        def _speak():
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 100)
                voices = engine.getProperty("voices")
                engine.setProperty("voice", voices[0].id)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print("Speak error:", e)
        threading.Thread(target=_speak, daemon=True).start()


    def clear_fun(self):
        self.str=" "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def predict(self, test_image):
        white=test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model(white, training=False)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]
        
        ref_dist = self.distance(self.pts[0], self.pts[9])
        if ref_dist == 0: ref_dist = 1
        
        def norm_dist(p1, p2):
            return self.distance(p1, p2) / ref_dist

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0
                # print("00000")

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
                print("++++++++++++++++++")
                # print("00000")

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2
                # print("22222")

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if norm_dist(self.pts[8], self.pts[16]) < 0.2:
                ch1 = 2
                # print("22222")


        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3
                print("33333c")



        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3
                print("33333b")

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3
                print("33333a")

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if norm_dist(self.pts[4], self.pts[11]) > 0.25:
                ch1 = 4
                # print("44444")

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (norm_dist(self.pts[4], self.pts[11]) > 0.23) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4
                # print("44444")

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4
                # print("44444")

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4
                # print("44444")

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4
                # print("44444")

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5
                print("55555b")

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5
                print("55555a")

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5
                # print("55555")

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5
                # print("55555")

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7
                # print("77777")

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7
                # print("77777")

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7
                # print("77777")

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6
                print("666661")


        # condition for [yj][x]
        print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6
                print("666662")

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if norm_dist(self.pts[8], self.pts[16]) > 0.23:
                ch1 = 6
                print("666663")

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if norm_dist(self.pts[4], self.pts[11]) < 0.28:
                ch1 = 6
                print("666664")

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6
                print("666665")

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1
                print("111111")

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
                print("111112")

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
                print("111112")

        # con for [d][pqz]
        fg = 19
        # print("_________________ch1=",ch1," ch2=",ch2)
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1
                print("111113")

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (norm_dist(self.pts[4], self.pts[11]) < 0.23) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1
                print("1111993")

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1
                print("1111mmm3")

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1
                print("1111140")

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1
                print("111114")

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7
                print("111114lll;;p")

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1
                print("111115")

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and norm_dist(self.pts[4], self.pts[11]) < 0.23:
                ch1 = 1
                print("111116")

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1
                print("1117")

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        #
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if norm_dist(self.pts[12], self.pts[4]) > 0.19:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if norm_dist(self.pts[8], self.pts[12]) > 0.33:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if norm_dist(self.pts[8], self.pts[4]) > 0.19:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if (norm_dist(self.pts[8], self.pts[12]) - norm_dist(self.pts[6], self.pts[10]) < 0.04) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if (norm_dist(self.pts[8], self.pts[12]) - norm_dist(self.pts[6], self.pts[10]) >= 0.04) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'

        if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1=" "



        # Keyboard handling replaces next gesture

        if ch1 == self.stable_pred:
            self.stable_counter += 1
        else:
            self.stable_pred = ch1
            self.stable_counter = 1

        if self.stable_counter >= 5:
            self.prev_char = ch1
            self.current_symbol = ch1
            self.count += 1
            self.ten_prev_char[self.count%10] = ch1
            self.stable_counter = 0


        if len(self.str.strip())!=0:
            st=self.str.rfind(" ")
            ed=len(self.str)
            word=self.str[st+1:ed]
            self.word=word
            print("----------word = ",word)
            if len(word.strip())!=0:
                suggestions = ddd.suggest(word)
                self.word1 = suggestions[0] if len(suggestions) >= 1 else " "
                self.word2 = suggestions[1] if len(suggestions) >= 2 else " "
                self.word3 = suggestions[2] if len(suggestions) >= 3 else " "
                self.word4 = suggestions[3] if len(suggestions) >= 4 else " "
            else:
                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "


    def destructor(self):

        print("Closing Application...")
        print(self.ten_prev_char)
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")

(Application()).root.mainloop()
