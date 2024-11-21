from lm import LanguageModel

print("Hi! This program will help you build an n-gram language model with an n (1-5) that you pick, train it on a dataset and generate random, new sentences from it. All you need is a text file to get started.")
trained = False

while True:
    if trained:
        choice = input("\nEnter 't' to train the language model on a corpus or 'g' to generate text(s) from a trained language model. Type 'e' to exit. > ")
    else:
        choice = input("\nEnter 't' to train the language model on a corpus. Type 'e' to exit. > ")

    if choice.lower() == 'e':
        break
    elif choice.lower() == 't':
        ng = input("\nEnter the number of tokens (n) to concatenate in an n-gram. Type 'b' to go back. > ")
        if ng.lower() == 'b':
            continue
        try:
            ng = int(ng)
            if ng <= 1:
                print("\nInvalid entry. Enter an integer greater than 1.")
                continue
            file = input("\nEnter the filename for the corpus you want to train the language model on. Type 'b' to go back. > ")
            if file.lower() == 'b':
                continue
            try:
                print("\nTraining...")
                lm = LanguageModel(ng)
                lm.train(file)
                trained = True
                print("\nTraining complete!")
            except FileNotFoundError:
                print("\nInvalid entry. Enter the name of an existing text file in your folder.")
        except ValueError:
            print("\nInvalid entry. Enter an integer greater than 1.")
    elif choice.lower() == 'g':
        if not trained:
            print("\nInvalid entry.")
            continue
        """
        Printing and writing options.
        """
        choice = input("\nEnter 'p' to print a text to the screen or enter 'w' to write text(s) to a file. Type 'b' to go back. > ")
        if choice.lower() == 'b':
            continue
        elif choice.lower() == 'p':
            ns = 9 if ng != 2 else 2
            text = ' '.join([lm.generate() for _ in range(ns)])
            # text = '\n' + ' '.join(lm.generate() for _ in range(ns)) + ' ' + lm.generate()
            text = text.replace(' .', '.')
            print(text)
        elif choice.lower() == 'w':
            while True:
                file = input('\nEnter the filename to write the generated text(s) to. Type \'b\' to go back. > ')
                if file.lower() == 'b':
                    break
                try:
                    num_texts = int(input('Enter the number of paragraphs to generate: '))
                    ns = 9 if ng != 2 else 2
                    lm.write_to_file(lm.generate(),file, num_texts, ns)
                    break
                except ValueError:
                    print('\nInvalid entry. Enter an integer.')
