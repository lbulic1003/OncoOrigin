# Library loading and relative path definition

import tkinter as tk
import pandas as pd
import xgboost as xgb
import os
import sys

base_path = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))

# Loading of machine learning model

kernel = pd.read_pickle(os.path.join(base_path, "kernel/kernel.pkl"))

# Loading of class dictionary and gene feature list

dictionary = pd.read_pickle(os.path.join(base_path, "misc/dictionary.pkl"))
gene_list = pd.read_pickle(os.path.join(base_path, "misc/gene_list.pkl"))

# Frame definition and icon loading

root = tk.Tk()
root.title("OncoOrigin v1.0.0")
root.geometry("850x500")
root.resizable(False, False)
root.iconbitmap(os.path.join(base_path, "img/logo3.ico"))

canvas = tk.Canvas(root, width=850, height=500, scrollregion=(0, 0, 850, 2300))
main = tk.Frame(canvas)
canvas.create_window(0, 0, window=main, anchor="nw")
scroll = tk.Scrollbar(root, orient="vertical")
canvas.config(yscrollcommand=scroll.set)
scroll.config(command=canvas.yview)
scroll.pack(fill="y", side="right")
canvas.pack(side="left")

# Definition of variables for input and output fields

prediction_vector = pd.DataFrame(index=[0], columns=(["Sex", "Age"] + gene_list))
ErrMsg = tk.StringVar()
prediction_var = tk.StringVar()
confidence_var = tk.StringVar()
probability1_var = tk.StringVar()
probability2_var = tk.StringVar()
probability3_var = tk.StringVar()
probability4_var = tk.StringVar()
probability5_var = tk.StringVar()
probability6_var = tk.StringVar()
probability7_var = tk.StringVar()
probability8_var = tk.StringVar()
probability9_var = tk.StringVar()
age_var = tk.StringVar()
sex_var = tk.IntVar()

gene_var_dict = {}
for i in gene_list:
    gene_var_dict[i] = tk.IntVar()

# Title and error bar

tk.Label(main, text="OncoOrigin", font=("Book Antiqua", 20, "bold"), fg="dark blue").grid(row=0, column=1, ipadx=55, ipady=5, columnspan=10, sticky="w")
ErrorBar = tk.Label(main, textvariable=ErrMsg, font=("Times New Roman", 10, "bold"), fg="red")
ErrorBar.grid(row=1, column=0, columnspan=10, ipadx=30, sticky="W")

# Patient age

tk.Label(main, text="Patient age (y) [18-89]:", font=("Times New Roman", 10)).grid(row=2, column=0, ipadx=30, sticky="W")
age_field = tk.Entry(main, textvariable=age_var, width=10)
age_field.grid(row=2, column=1, sticky="W")

# Patient sex

tk.Label(main, text="Patient sex:", font=("Times New Roman", 10)).grid(row=3, column=0, ipadx=30, sticky="W")
male_button = tk.Radiobutton(main, text="Male", variable=sex_var, value=0)
male_button.grid(row=3, column=1, sticky="W")
female_button = tk.Radiobutton(main, text="Female", variable=sex_var, value=1)
female_button.grid(row=3, column=2, sticky="W")
sex_var.set(0)

# Gene variants

int_row = 4
tk.Label(main, text="Detected genetic variants:", font=("Book Antiqua", 10)).grid(row=4, column=0, ipadx=30, sticky="W")
for i in range(len(gene_list)):
    int_row = i//5 + 4
    tk.Checkbutton(main, text=gene_list[i], variable=gene_var_dict[gene_list[i]], onvalue=1, offvalue=0).grid(row=int_row, column=(i%5 + 1), sticky="W", ipadx=10)
    gene_var_dict[gene_list[i]].set(0)

tk.Label(main, text="", font=("Book Antiqua", 10)).grid(row=(int_row+1), ipadx=30, ipady=5)

# Main buttons

predict_button = tk.Button(main, text="Predict", background="light green")
predict_button.grid(row=(int_row+2), column=1, columnspan=3, ipadx=30, sticky="W")
reset_button = tk.Button(main, text="Reset", background="pink")
reset_button.grid(row=(int_row+2), column=3, columnspan=3, ipadx=30, sticky="W")

tk.Label(main, text="", font=("Book Antiqua", 10)).grid(row=(int_row+3), ipadx=30, ipady=5)

# Prediction, confidence and relative probabilities

tk.Label(main, text="Predicted primary:", font=("Book Antiqua", 15)).grid(row=(int_row+4), column=0, ipadx=30, sticky="W")
PredictionBar = tk.Label(main, textvariable=prediction_var, font=("Book Antiqua", 15, "bold"))
PredictionBar.grid(row=(int_row+4), column=1, columnspan=9, ipadx=30, sticky="W")
tk.Label(main, text="Confidence level:", font=("Book Antiqua", 11)).grid(row=(int_row+5), column=0, ipadx=30, sticky="W")
ConfidenceBar = tk.Label(main, textvariable=confidence_var, font=("Book Antiqua", 11, "bold"))
ConfidenceBar.grid(row=(int_row+5), column=1, columnspan=9, ipadx=30, sticky="W")
tk.Label(main, text="Probabilities relative to predicted class:", font=("Book Antiqua", 11)).grid(row=(int_row+6), column=0, ipadx=30, sticky="W")
ProbabilityBar1 = tk.Label(main, textvariable=probability1_var, font=("Book Antiqua", 11, "bold"))
ProbabilityBar1.grid(row=(int_row+6), column=1, columnspan=30, ipadx=30, sticky="W")
ProbabilityBar2 = tk.Label(main, textvariable=probability2_var, font=("Book Antiqua", 11))
ProbabilityBar2.grid(row=(int_row+7), column=1, columnspan=30, ipadx=30, sticky="W")
ProbabilityBar3 = tk.Label(main, textvariable=probability3_var, font=("Book Antiqua", 11))
ProbabilityBar3.grid(row=(int_row+8), column=1, columnspan=30, ipadx=30, sticky="W")
ProbabilityBar4 = tk.Label(main, textvariable=probability4_var, font=("Book Antiqua", 11))
ProbabilityBar4.grid(row=(int_row+9), column=1, columnspan=30, ipadx=30, sticky="W")
ProbabilityBar5 = tk.Label(main, textvariable=probability5_var, font=("Book Antiqua", 11))
ProbabilityBar5.grid(row=(int_row+10), column=1, columnspan=30, ipadx=30, sticky="W")
ProbabilityBar6 = tk.Label(main, textvariable=probability6_var, font=("Book Antiqua", 11))
ProbabilityBar6.grid(row=(int_row+11), column=1, columnspan=30, ipadx=30, sticky="W")
ProbabilityBar7 = tk.Label(main, textvariable=probability7_var, font=("Book Antiqua", 11))
ProbabilityBar7.grid(row=(int_row+12), column=1, columnspan=30, ipadx=30, sticky="W")
ProbabilityBar8 = tk.Label(main, textvariable=probability8_var, font=("Book Antiqua", 11))
ProbabilityBar8.grid(row=(int_row+13), column=1, columnspan=30, ipadx=30, sticky="W")
ProbabilityBar9 = tk.Label(main, textvariable=probability9_var, font=("Book Antiqua", 11))
ProbabilityBar9.grid(row=(int_row+14), column=1, columnspan=30, ipadx=30, sticky="W")

# Main functions

# Reset - resets the entire window

def reset():
    ConfidenceBar.configure(fg="black") 
    prediction_var.set("")
    confidence_var.set("")
    probability1_var.set("")
    probability2_var.set("")
    probability3_var.set("")
    probability4_var.set("")
    probability5_var.set("")
    probability6_var.set("")
    probability7_var.set("")
    probability8_var.set("")
    probability9_var.set("")
    age_var.set("")
    ErrMsg.set("")
    sex_var.set(0)
    for i in range(len(gene_list)):
        gene_var_dict[gene_list[i]].set(0)
    prediction_vector.drop(prediction_vector.index, inplace=True)
    return

# Predict - calculates and outputs prediction, confidence and relative probabilities

def predict():

    prediction_vector.drop(prediction_vector.index, inplace=True)
    prediction_list = []
    ErrMsg.set("")
    age = age_var.get()
    sex = sex_var.get()
    try:
        age = int(age)
        if(age<18 or age>89):
            raise ValueError
    except ValueError:
        reset()
        ErrMsg.set("INCORRECT AGE VALUE")
        return
    prediction_list.append(sex)
    prediction_list.append(age)
    for i in gene_list:
        prediction_list.append(gene_var_dict[i].get())
    prediction_vector.loc[0] = prediction_list
    prediction = kernel.predict(prediction_vector)
    probabilities = kernel.predict_proba(prediction_vector)

    configure = [0, 0, 0, 0]

    max_prob=0
    max_prob_ind=0
    for i in range(10):
        if(probabilities[0, i]>max_prob):
            max_prob = probabilities[0, i]
            max_prob_ind = i

    rel_prob = {}

    for i in range(10):
        prob = probabilities[0, i]/max_prob
        rel_prob[prob] = i
        if(i!=max_prob_ind):
            if(prob >= 0.8):
                configure[3] = 1
            elif(prob < 0.8 and prob >= 0.6):
                configure[2] = 1
            elif(prob < 0.6 and prob >= 0.4):
                configure[1] = 1
            elif(prob < 0.4 and prob >= 0.2):
                configure[0] = 1

    if(configure[3]==1):
        confidence_var.set("Very low")
        ConfidenceBar.configure(fg="#FF0000")
    elif(configure[2]==1):
        confidence_var.set("Low")
        ConfidenceBar.configure(fg="#FF7900")
    elif(configure[1]==1):
        confidence_var.set("Moderate")
        ConfidenceBar.configure(fg="#E4D00A")
    elif(configure[0]==1):
        confidence_var.set("High")
        ConfidenceBar.configure(fg="#32CD32")
    else:
        confidence_var.set("Very high")
        ConfidenceBar.configure(fg="#2E8B57")        

    prediction_var.set(dictionary[prediction[0]])

    key_order = sorted(rel_prob.keys(), reverse=True)

    probability1_var.set(f"{dictionary[rel_prob[key_order[1]]]}, {key_order[1]:.1f}")
    probability2_var.set(f"{dictionary[rel_prob[key_order[2]]]}, {key_order[2]:.1f}")
    probability3_var.set(f"{dictionary[rel_prob[key_order[3]]]}, {key_order[3]:.1f}")
    probability4_var.set(f"{dictionary[rel_prob[key_order[4]]]}, {key_order[4]:.1f}")
    probability5_var.set(f"{dictionary[rel_prob[key_order[5]]]}, {key_order[5]:.1f}")
    probability6_var.set(f"{dictionary[rel_prob[key_order[6]]]}, {key_order[6]:.1f}")
    probability7_var.set(f"{dictionary[rel_prob[key_order[7]]]}, {key_order[7]:.1f}")
    probability8_var.set(f"{dictionary[rel_prob[key_order[8]]]}, {key_order[8]:.1f}")
    probability9_var.set(f"{dictionary[rel_prob[key_order[9]]]}, {key_order[9]:.1f}")

    return

predict_button.configure(command=predict)
reset_button.configure(command=reset)

# Disclaimer

tk.Label(main, text="", font=("Book Antiqua", 10)).grid(row=(int_row+15), column=0, ipady=5)
tk.Label(main, text="This version of OncoOrigin is experimental and is not intended for clinical use.\nThe creators are not liable for any clinical decisions made based on this version of the software.", font=("Book Antiqua", 8)).grid(row=(int_row+16), column=0, ipadx=70, columnspan=60, ipady=5)

root.mainloop()