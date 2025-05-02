from assistant.interactive_agent import interactive_agent


if __name__ == "__main__":
    username = "admin"
    user_agent, query_engine = interactive_agent(username)

    if user_agent and query_engine:
        # Interactive chat loop at the higher level
        while True:
            user_input = input("\n>> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break

            # Process query through the UserQueryAgent
            response = user_agent.process_query(user_input, query_engine)
            
            # If the response is None, it means the command output was already printed
            # or the command wasn't handled and should be processed by the default logic
            if response is not None:
                print(f"Response: {response}")
    else:
        print("No valid report directories found. Please check the reports directory.")
    