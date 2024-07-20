from colorama import Fore, Back, Style, init
init(autoreset=True)

def chatbot():
    welcome_message = """
    ***********************************
    *                                 *
    *       WELCOME TO CHATBOT        *
    *                                 *
    ***********************************
    """
    print(Fore.GREEN + welcome_message)

    while True:
        user_input = input(Fore.YELLOW + "You: ").lower()

        if user_input in ["hi", "hello", "hey"]:
            response = Fore.CYAN + "Chatbot: Hello! How can I assist you today?"
        elif "name" in user_input:
            response = Fore.CYAN + "Chatbot: I'm a chatbot created to assist you with simple queries."
        elif "help" in user_input:
            response = Fore.CYAN + "Chatbot: Sure, I'm here to help. What do you need assistance with?"
        elif "bye" in user_input or "exit" in user_input:
            response = Fore.CYAN + "Chatbot: Goodbye! Have a great day!"
            print(response)
            break
        else:
            response = Fore.CYAN + "Chatbot: I'm sorry, I don't understand that. Can you please rephrase?"

        print(response)

if __name__ == "__main__":
    chatbot()
