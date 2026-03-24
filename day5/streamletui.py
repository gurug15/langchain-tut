import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage




config = {"configurable":{"thread_id":"thread-1"}}


if 'message_history' not in  st.session_state:
    st.session_state['message_history']= []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input('type here')

if user_input:
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=config)

    ai_message = response['messages'][-1].content
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)
# import { useState } from "react";

# import Markdown from "react-markdown";
# import remarkGfm from "remark-gfm";

# function App() {
#   const [messages, setMessages] = useState([]);

#   const sendMessage = async (input) => {
#     // Add user message
#     setMessages((prev) => [...prev, { role: "user", content: input }]);

#     // Add empty assistant message
#     let assistantMessage = { role: "assistant", content: "" };
#     setMessages((prev) => [...prev, assistantMessage]);

#     const response = await fetch("http://localhost:5555/chat", {
#       method: "POST",
#       headers: {
#         "Content-Type": "application/json",
#       },
#       body: JSON.stringify({ message: input }),
#     });

#     const reader = response.body.getReader();
#     const decoder = new TextDecoder("utf-8");

#     let done = false;
#     let accumulated = "";

#     while (!done) {
#       const { value, done: doneReading } = await reader.read();
#       done = doneReading;

#       const chunk = decoder.decode(value || new Uint8Array());

#       accumulated += chunk;

#       // Update last assistant message
#       setMessages((prev) => {
#         const updated = [...prev];
#         updated[updated.length - 1] = {
#           role: "assistant",
#           content: accumulated,
#         };
#         return updated;
#       });
#     }
#   };

#   return (
#     <div style={{ padding: 20 }}>
#       <h2>Chat</h2>

#       <div>
#         {messages.map((msg, i) => (
#           <div key={i}>
#             <b>{msg.role}:</b>{" "}
#             <Markdown remarkPlugins={[remarkGfm]}>{msg.content}</Markdown>
#           </div>
#         ))}
#       </div>

#       <input
#         type="text"
#         placeholder="Type message..."
#         onKeyDown={(e) => {
#           if (e.key === "Enter") {
#             sendMessage(e.target.value);
#             e.target.value = "";
#           }
#         }}
#       />
#     </div>
#   );
# }

# export default App;
