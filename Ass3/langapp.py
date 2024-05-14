print("LOADING...")
import os
import warnings
import pandas as pd

from platform import system
from random import choice
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")


# Function for clearing the screen
clear = lambda: os.system('cls' if system() == "Windows" else 'clear')

def menu():
    print("-----------------MENU-----------------")
    print("(1) Get random word")
    print("(2) Practice vocabulary")
    print("(3) Word selection game")
    print("(0) Quit\n")
    
def get_and_validate_input(valid, current_screen, *screen_args):
    while True:
        inp = input()
        if inp in valid:
            return int(inp)
        # If the user inout is invalid, reprint the current screen
        clear()
        current_screen(*screen_args)

# This ensures that the probabilities are reset for each game
def init_words(words):
    df = words.copy()
    df['prob'] = len(df)
    return df

# The selected word(s) are reset to the minimal probability
# The rest become comparatively more likely to be selected next
def update_probs(words, selection):
    words.loc[selection.index, 'prob'] = 0
    words.prob += len(selection)

# ---------The random word screen---------
def get_random(words):
    while True:
        clear()
        word = words.sample(1, weights='prob')
        random_menu(word)
        inp = get_and_validate_input(['0', '1'], random_menu, word)
        if inp == 0: break
        else: update_probs(words, word)
        
def random_menu(word):
    print(f'Your random word is "{word.word.values[0]}"')
    print(f'It is defined as "{word.definition.values[0]}"')
    print()
    print("(1) Get a new random word")
    print("(0) Return\n")

# ---------The term/definition matching screen---------
def vocabulary(words):
    clear()
    vocabulary_select_menu()
    inp = get_and_validate_input(['0', '1', '2', '3'], vocabulary_select_menu)
    if inp != 0: vocabulary_game(inp, words)

def vocabulary_select_menu():
    print("Practice by writing down:")
    print("(1) Definitions based on words")
    print("(2) Words based on definitions")
    print("(3) Mix of both")
    print("(0) Return\n")

def vocabulary_game(mode, words):
    random = mode == 3
    while True:
        clear()
        word = words.sample(1, weights='prob')
        if random: mode = choice([1, 2])
        vocabulary_game_menu1(word, mode)
        inp = input()
        vocabulary_game_menu2(inp, word, mode)
        menu = lambda: (vocabulary_game_menu1(word, mode), print(inp), vocabulary_game_menu2(inp, word, mode))
        inp = get_and_validate_input(['0', '1'], menu)
        if inp == 0: break
        else: update_probs(words, word)

def vocabulary_game_menu1(word, mode):
    if mode == 1:
        print(f'What is the definition of the word "{word.word.values[0]}"')
    else:
        print(f'What word matches the definition "{word.definition.values[0]}"')

def vocabulary_game_menu2(inp, word, mode):
    if mode == 1:
        pred, real = model.encode([inp]), model.encode(word.definition.values)
        score = cosine_similarity(real, pred) / (norm(real - pred) + 1e-8)
        print("Correct!" if score > 0.5 else "Incorrect!")
        print(f"The right answer was: {word.definition.values[0]}")
    else:
        print("Correct!" if word.word.values[0].lower() == inp.lower() else "Incorrect!")
        print(f"The right answer was: {word.word.values[0]}")
    print(f"You answered: {inp}")
    print()
    print("(1) Next word")
    print("(0) Return\n")

# ---------The Selection game screen---------
def selection_game(words):
    while True:
        clear()
        selected = words.sample(2, weights='prob')
        ans = choice([0, 1])
        selection_menu1(selected, ans)
        inp = get_and_validate_input(['1', '2'], selection_menu1, selected, ans)
        selection_menu2(ans == inp - 1)
        menu = lambda: (selection_menu1(selected, ans), print(inp), selection_menu2(ans == inp - 1))
        inp = get_and_validate_input(['0', '1'], menu)
        if inp == 0: break
        else: update_probs(words, words)

def selection_menu1(words, ans):
    print(f'Which word matches the definition "{words.iloc[ans].definition}"')
    print(f"(1) {words.iloc[0].word}")
    print(f"(2) {words.iloc[1].word}")

def selection_menu2(correct):
    print("Correct!" if correct else "Incorrect!")
    print()
    print("(1) Next word")
    print("(0) Return\n")

selection = {
    1: get_random,
    2: vocabulary,
    3: selection_game
}


words = pd.read_csv('words.txt', sep=' - ', header=None, engine='python', names=['word', 'definition'])
model = SentenceTransformer("all-MiniLM-L6-v2")

clear()
while True:
    menu()
    inp = get_and_validate_input(['0', '1', '2', '3'], menu)
    if inp == 0: break
    selection[inp](init_words(words))
    clear()