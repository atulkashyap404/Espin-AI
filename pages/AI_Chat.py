import streamlit as st
from utils.api import fetch_groq_response

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧠 Nanofiber AI Chat")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Ask about nanofibers...")

if user_input:
    # Add user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Fetch AI response
    response = fetch_groq_response(user_input)
    
    # Add AI response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display updated chat history
    st.rerun()

# 🔹 Reset Chat Button
if st.button("🔄 Reset Chat"):
    st.session_state.chat_history = []  # Clears chat history
    st.rerun()  # Refresh the page to reset chat











# import streamlit as st
# from utils.api import fetch_groq_response

# # Initialize chat history in session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# st.title("🧠 Nanofiber AI Chat")

# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

#         # If it's an assistant's response, show the "Copy" icon button
#         if message["role"] == "assistant":
#             st.code(message["content"], language="markdown")  # Proper formatting
#             st.button("📋 Copy", key=message["content"], on_click=lambda: st.session_state.update({"copy_text": message["content"]}))

# # Chat input
# user_input = st.chat_input("Ask about nanofibers...")

# if user_input:
#     # Add user input to chat history
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
    
#     # Fetch AI response
#     response = fetch_groq_response(user_input)
    
#     # Add AI response to chat history
#     st.session_state.chat_history.append({"role": "assistant", "content": response})

#     # Display updated chat history
#     st.rerun()

# # 🔹 Reset Chat Button
# if st.button("🔄 Reset Chat"):
#     st.session_state.chat_history = []  # Clears chat history
#     st.rerun()  # Refresh the page to reset chat











# import streamlit as st
# from utils.api import fetch_groq_response

# # Initialize chat history in session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# st.title("Nanofiber AI Chat")

# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # Chat input
# user_input = st.chat_input("Ask about nanofibers...")

# if user_input:
#     # Add user input to chat history
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
    
#     # Fetch AI response
#     response = fetch_groq_response(user_input)
    
#     # Add AI response to chat history
#     st.session_state.chat_history.append({"role": "assistant", "content": response})

#     # Display updated chat history
#     st.rerun()

# # 🔹 Add Reset Chat Button
# if st.button("Reset Chat"):
#     st.session_state.chat_history = []  # Clears chat history
#     st.rerun()  # Refresh the page to reset chat









# import streamlit as st
# from utils.api import fetch_groq_response

# st.title("🤖 Nanofiber AI Chat")
# st.subheader("Ask anything about nanofibers!")

# # Chat history
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# # Display chat history
# for msg in st.session_state["messages"]:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # User input
# user_input = st.chat_input("Type your question...")
# if user_input:
#     st.session_state["messages"].append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)
    
#     # Get AI response
#     ai_response = fetch_groq_response(user_input)
#     st.session_state["messages"].append({"role": "assistant", "content": ai_response})
#     with st.chat_message("assistant"):
#         st.markdown(ai_response)
