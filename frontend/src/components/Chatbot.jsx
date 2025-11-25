import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { Send, Bot, User, Phone, X, MessageSquare } from "lucide-react";

// --- CSS for Animations ---
const GlobalStyles = () => (
  <style jsx global>{`
    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .message-fade-in {
      animation: fadeIn 0.3s ease-out;
    }

    @keyframes bounce {
      0%,
      80%,
      100% {
        transform: scale(0);
      }
      40% {
        transform: scale(1);
      }
    }

    .typing-dot {
      animation: bounce 1.4s infinite ease-in-out both;
    }

    .typing-dot:nth-child(1) {
      animation-delay: -0.32s;
    }

    .typing-dot:nth-child(2) {
      animation-delay: -0.16s;
    }
  `}</style>
);

// --- Typing Indicator Component ---
const TypingIndicator = () => (
  <div className="flex items-center gap-2.5 p-4">
    <div className="flex items-center justify-center w-8 h-8 bg-gray-200 rounded-full flex-shrink-0">
      <Bot className="w-5 h-5 text-gray-600" />
    </div>
    <div className="flex items-center space-x-1.5 bg-gray-200 px-4 py-3 rounded-xl rounded-bl-none shadow-sm">
      <div className="w-2 h-2 bg-gray-500 rounded-full typing-dot"></div>
      <div className="w-2 h-2 bg-gray-500 rounded-full typing-dot"></div>
      <div className="w-2 h-2 bg-gray-500 rounded-full typing-dot"></div>
    </div>
  </div>
);

// --- Main Chatbot Component ---
const Chatbot = () => {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "👋 Hi! How can I assist you with your university queries today?" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showTelecallerPrompt, setShowTelecallerPrompt] = useState(false);
  const [connectionError, setConnectionError] = useState(false);
  const chatAreaRef = useRef(null);

  // Backend URL - CHANGE THIS TO MATCH YOUR FLASK SERVER
  const BACKEND_URL = "http://localhost:5000"; // Flask runs on port 5000 by default

  // Scroll to bottom whenever messages change
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages, loading]);

  // Test backend connection on component mount
  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/health`);
      console.log("✅ Backend connection successful:", response.data);
      setConnectionError(false);
    } catch (error) {
      console.error("❌ Backend connection failed:", error);
      setConnectionError(true);
      setMessages(prev => [
        ...prev,
        { sender: "bot", text: "⚠️ Backend server is not running. Please start the Flask server on port 5000." }
      ]);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setShowTelecallerPrompt(false);

    try {
      const response = await axios.post(`${BACKEND_URL}/api/chat/message`, {
        message: input,
      });

      console.log("🔍 FRONTEND - Full API Response:", response.data);

      if (response.data.success) {
        const data = response.data.data;
        
        // Use the response from the API
        const botResponse = data.response;

        // Check if telecaller is required using requiresFollowup
        const shouldShowTelecaller = data.requiresFollowup === true;

        const botMsg = { sender: "bot", text: botResponse };

        setTimeout(() => {
          setMessages((prev) => [...prev, botMsg]);
          setLoading(false);

          // Show telecaller prompt if requires followup
          if (shouldShowTelecaller) {
            console.log("📞 Showing telecaller prompt - requiresFollowup:", data.requiresFollowup);
            setShowTelecallerPrompt(true);
          }
        }, 750);
      } else {
        throw new Error(response.data.error || 'Unknown error');
      }

    } catch (err) {
      setLoading(false);
      setConnectionError(true);
      setMessages((prev) => [
        ...prev,
        { 
          sender: "bot", 
          text: "❌ Sorry, I'm having trouble connecting to the server. Please make sure the backend is running on port 5000." 
        },
      ]);
      console.error("API Error:", err);
    }
  };

  const handleTelecallerAction = async (choice) => {
    setShowTelecallerPrompt(false);
    
    if (choice === "yes") {
      const userMessage = messages[messages.length - 2]?.text || "General inquiry";
      
      try {
        // Send telecaller request to backend
        await axios.post(`${BACKEND_URL}/api/telecaller/request`, {
          user_message: userMessage,
          contact: "User requested callback"
        });

        setMessages((prev) => [
          ...prev,
          { sender: "user", text: "Yes, please connect me with a telecaller." },
          {
            sender: "bot",
            text: "☎️ Great! Our university telecaller will reach out to you shortly. They can provide more detailed information and assistance. Is there anything else I can help you with in the meantime?",
          },
        ]);
      } catch (error) {
        setMessages((prev) => [
          ...prev,
          { sender: "user", text: "Yes, please connect me with a telecaller." },
          {
            sender: "bot",
            text: "☎️ I've noted your request for a telecaller. Our team will contact you soon. Is there anything else I can help with?",
          },
        ]);
      }
    } else {
      setMessages((prev) => [
        ...prev,
        { sender: "user", text: "No, thank you. I'll continue with the chatbot." },
        {
          sender: "bot",
          text: "👍 Understood! Feel free to ask me any other questions about Adamas University. I'm here to help!",
        },
      ]);
    }
  };

  return (
    <>
      <GlobalStyles />
      <div className="w-full max-w-md h-[700px] bg-gray-100 shadow-2xl rounded-2xl flex flex-col overflow-hidden border border-gray-200">

        {/* Header */}
        <div className="bg-white text-gray-800 py-4 px-5 flex items-center justify-between border-b border-gray-200 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-100 rounded-full">
              <MessageSquare className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Adamas University Chatbot</h3>
              <p className={`text-xs font-medium ${connectionError ? 'text-red-500' : 'text-green-500'}`}>
                {connectionError ? '● Connection Issue' : '● Online'}
              </p>
            </div>
          </div>
        </div>

        {/* Chat Area */}
        <div
          ref={chatAreaRef}
          className="flex-1 overflow-y-auto p-4 space-y-4 bg-white"
        >
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex items-start gap-3 message-fade-in ${msg.sender === "user" ? "justify-end" : "justify-start"
                }`}
            >
              {/* Bot Avatar */}
              {msg.sender !== "user" && (
                <div className="flex items-center justify-center w-8 h-8 bg-gray-200 rounded-full flex-shrink-0 shadow-sm">
                  <Bot className="w-5 h-5 text-gray-600" />
                </div>
              )}

              {/* Message Bubble */}
              <div
                className={`relative max-w-[75%] px-4 py-3 rounded-xl shadow-sm ${msg.sender === "user"
                    ? "bg-blue-600 text-white rounded-br-none"
                    : "bg-gray-200 text-gray-800 rounded-bl-none"
                  }`}
              >
                {msg.text}
              </div>

              {/* User Avatar */}
              {msg.sender === "user" && (
                <div className="flex items-center justify-center w-8 h-8 bg-blue-100 rounded-full flex-shrink-0 shadow-sm">
                  <User className="w-5 h-5 text-blue-600" />
                </div>
              )}
            </div>
          ))}

          {/* Typing indicator */}
          {loading && <TypingIndicator />}
        </div>

        {/* Telecaller Prompt Area */}
        {showTelecallerPrompt && (
          <div className="py-3 px-4 bg-yellow-50 border-t border-yellow-200 message-fade-in">
            <p className="text-sm text-center text-gray-700 mb-2 font-medium">
              📞 Would you like to speak with our university telecaller for personalized assistance?
            </p>
            <div className="flex justify-center gap-3">
              <button
                onClick={() => handleTelecallerAction("yes")}
                className="flex-1 flex items-center justify-center gap-1.5 bg-green-500 hover:bg-green-600 text-white px-3 py-2 rounded-lg text-sm font-semibold transition-all shadow-sm"
              >
                <Phone className="w-4 h-4" />
                Yes, Connect
              </button>
              <button
                onClick={() => handleTelecallerAction("no")}
                className="flex-1 flex items-center justify-center gap-1.5 bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 px-3 py-2 rounded-lg text-sm font-semibold transition-all shadow-sm"
              >
                <X className="w-4 h-4" />
                No, Thanks
              </button>
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="border-t border-gray-200 flex items-center gap-3 p-4 bg-gray-50">
          <input
            type="text"
            className="flex-1 border border-gray-300 rounded-full px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="Ask about admissions, fees, hostel, timings..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !loading && !showTelecallerPrompt && handleSend()}
            disabled={loading || showTelecallerPrompt}
          />
          <button
            onClick={handleSend}
            className="bg-blue-600 text-white p-2.5 rounded-full hover:bg-blue-700 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={loading || showTelecallerPrompt || !input.trim()}
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </>
  );
};

export default Chatbot;