import numpy as np

# TODO [1]: implement the guessing_game function
def guessing_game(max: int, *, attempts: int): # hint: return type is tuple[bool, list[int], int]:
    chosen_int = np.random.randint(low=1, high=max+1)
    guesses = []

    print(f"Welcome to the Guessing Game! Guess a number between 1 and {max}. You have {attempts} attempts.")
    for attempt in range(attempts):
        try:
            guess = int(input(f"Attempt {attempt + 1}/{attempts}: Enter your guess: "))
            guesses.append(guess)

            if guess < chosen_int:
                print(f"Too low! You have {attempts - attempt - 1} attempts left.")
            elif guess > chosen_int:
                print(f"Too high! You have {attempts - attempt - 1} attempts left.")
            else:
                print(f"Congratulations! You guessed the correct number: {chosen_int}.")
                return True, guesses, chosen_int
        except ValueError:
            print("Invalid input! Please enter a valid integer.")

    print(f"Sorry, you've run out of attempts. The correct number was {chosen_int}.")
    return False, guesses, chosen_int
    
    pass

# TODO [2]: implement the play_game function
def play_game()-> None:
    max_value:int = 20
    attempts:int = 5
    
    while True:
        result, guesses, chosen_int = guessing_game(max_value, attempts=attempts)

        if result:
            # Assert that the chosen number is in the guesses list
            assert chosen_int in guesses, "Error: The chosen number is not in the guesses list."
            print("You won! Game over.")
            break
        else:
            # Assert that the chosen number is not in the guesses list
            assert chosen_int not in guesses, "Error: The chosen number is in the guesses list."
            print("You lost! The correct number was not in your guesses.")
            play_again = input("Do you want to play again? (yes/no): ").strip().lower()
            if play_again != "yes":
                print("Thanks for playing! Goodbye.")
                break
            

