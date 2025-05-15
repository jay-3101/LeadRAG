// Dashboard JavaScript - handles both user and admin interfaces
document.addEventListener('DOMContentLoaded', function() {
    // Get user data from the body
    const userType = document.body.getAttribute('data-user-type');
    const userEmail = document.body.getAttribute('data-email');
    
    // Get elements
    const formSelector = userType === 'admin' ? '#adminQueryForm' : '#userQueryForm';
    const form = document.querySelector(formSelector);
    const chatMessages = document.getElementById('chatMessages');
    const modelSelect = document.getElementById('modelSelect');
    const newConversationBtn = document.getElementById('newConversationBtn');
    const conversationList = document.getElementById('conversationList');
    const currentConversationName = document.getElementById('currentConversationName');
    
    // State management
    let currentConversationId = null;
    let conversations = {};
    
    // Initialize
    loadConversations();
    setupEventListeners();
    
    // Event listeners setup
    function setupEventListeners() {
      // New conversation button
      newConversationBtn.addEventListener('click', createNewConversation);
      
      // Form submission
      form.addEventListener('submit', function(e) {
        e.preventDefault();
        const textarea = this.querySelector('textarea');
        const query = textarea.value.trim();
        
        if (query) {
          sendQuery(query);
          textarea.value = '';
        }
      });
      
      // Upload form for admin
      if (userType === 'admin') {
        const uploadForm = document.getElementById('uploadForm');
        if (uploadForm) {
          uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            uploadDocument(this);
          });
        }
      }
    }
    
    // Load user conversations
    async function loadConversations() {
      try {
        const response = await fetch('/conversations');
        if (response.ok) {
          conversations = await response.json();
          
          // Clear conversation list
          const noConversationsEl = conversationList.querySelector('.no-conversations');
          conversationList.innerHTML = '';
          
          // Check if we have conversations
          if (Object.keys(conversations).length === 0) {
            conversationList.innerHTML = '<div class="no-conversations">No previous chats found</div>';
            createNewConversation();
            return;
          }
          
          // Add conversations to the list
          for (const [id, convo] of Object.entries(conversations)) {
            addConversationToList(id, convo);
          }
          
          // Open the most recent conversation
          const sortedIds = Object.keys(conversations).sort((a, b) => {
            return new Date(conversations[b].updated_at) - new Date(conversations[a].updated_at);
          });
          
          if (sortedIds.length > 0) {
            openConversation(sortedIds[0]);
          } else {
            createNewConversation();
          }
        } else {
          console.error('Failed to load conversations');
          createNewConversation();
        }
      } catch (error) {
        console.error('Error loading conversations:', error);
        createNewConversation();
      }
    }
    
    // Add a conversation to the sidebar list
    function addConversationToList(id, convo) {
      // Find the first user message as the title
      let title = 'New Conversation';
      let preview = 'No messages yet';
      
      if (convo.messages && convo.messages.length > 0) {
        // Find the first user message
        const firstUserMessage = convo.messages.find(msg => msg.role === 'user');
        if (firstUserMessage) {
          title = firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '');
        }
        
        // Get the last message as preview
        const lastMessage = convo.messages[convo.messages.length - 1];
        preview = `${lastMessage.role === 'user' ? 'You: ' : 'Assistant: '}${lastMessage.content.substring(0, 30)}${lastMessage.content.length > 30 ? '...' : ''}`;
      }
      
      // Format date
      const updatedAt = new Date(convo.updated_at).toLocaleString();
      
      // Create conversation item
      const conversationItem = document.createElement('div');
      conversationItem.className = 'conversation-item';
      conversationItem.dataset.id = id;
      conversationItem.innerHTML = `
        <div class="conversation-title">${title}</div>
        <div class="conversation-preview">${preview}</div>
      `;
      
      // Add click handler
      conversationItem.addEventListener('click', function() {
        openConversation(id);
      });
      
      // Add to list
      conversationList.appendChild(conversationItem);
    }
    
    // Open a conversation
    async function openConversation(id) {
      try {
        // If it's already in memory
        if (conversations[id]) {
          displayConversation(id, conversations[id]);
          return;
        }
        
        // Otherwise, fetch it
        const response = await fetch(`/conversations/${id}`);
        if (response.ok) {
          const conversation = await response.json();
          conversations[id] = conversation;
          displayConversation(id, conversation);
        } else {
          console.error('Failed to load conversation:', id);
        }
      } catch (error) {
        console.error('Error opening conversation:', error);
      }
    }
    
    // Create a new conversation
    async function createNewConversation() {
      try {
        const response = await fetch('/conversation/new', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            user_type: userType,
            email: userEmail,
            model: modelSelect.value
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          currentConversationId = data.conversation_id;
          
          // Create an empty conversation in memory
          conversations[currentConversationId] = {
            user_email: userEmail,
            user_type: userType,
            model_type: modelSelect.value,
            messages: [],
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          };
          
          // Display the new conversation
          displayConversation(currentConversationId, conversations[currentConversationId]);
          
          // Update the conversation list if there's no "New Conversation" item yet
          const noConvos = conversationList.querySelector('.no-conversations');
          if (noConvos) {
            conversationList.innerHTML = '';
          }
          
          addConversationToList(currentConversationId, conversations[currentConversationId]);
        } else {
          console.error('Failed to create new conversation');
        }
      } catch (error) {
        console.error('Error creating conversation:', error);
      }
    }
    
    // Display a conversation
    function displayConversation(id, conversation) {
      // Update current conversation
      currentConversationId = id;
      
      // Set model select
      if (conversation.model_type) {
        modelSelect.value = conversation.model_type;
      }
      
      // Update conversation name
      let title = 'New Conversation';
      if (conversation.messages && conversation.messages.length > 0) {
        const firstUserMessage = conversation.messages.find(msg => msg.role === 'user');
        if (firstUserMessage) {
          title = firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '');
        }
      }
      currentConversationName.textContent = title;
      
      // Highlight active conversation in sidebar
      document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.id === id) {
          item.classList.add('active');
        }
      });
      
      // Clear chat messages
      chatMessages.innerHTML = '';
      
      // If no messages, show welcome message
      if (!conversation.messages || conversation.messages.length === 0) {
        chatMessages.innerHTML = `
          <div class="welcome-message">
            <p>Hello${userType === 'admin' ? ' Admin' : ''}! How can I assist you today?</p>
          </div>
        `;
        return;
      }
      
      // Display all messages
      conversation.messages.forEach(message => {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`;
        messageEl.innerHTML = message.content.replace(/\n/g, '<br>');
        chatMessages.appendChild(messageEl);
      });
      
      // Scroll to bottom
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Send a query
    async function sendQuery(query) {
      try {
        // Add user message to the UI immediately
        addMessage('user', query);
        
        // If no active conversation, create one
        if (!currentConversationId) {
          await createNewConversation();
        }
        
        // Add user message to conversation in memory
        if (conversations[currentConversationId]) {
          if (!conversations[currentConversationId].messages) {
            conversations[currentConversationId].messages = [];
          }
          
          conversations[currentConversationId].messages.push({
            role: 'user',
            content: query,
            timestamp: new Date().toISOString()
          });
        }
        
        // Send to server
        const response = await fetch('/query', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            query: query,
            user_type: userType,
            email: userEmail,
            model: modelSelect.value,
            conversation_id: currentConversationId
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          
          // Add response to UI
          addMessage('assistant', data.response,data.context);
          
          // Add response to conversation in memory
          if (conversations[currentConversationId]) {
            conversations[currentConversationId].messages.push({
              role: 'assistant',
              content: data.response,
              timestamp: new Date().toISOString()
            });
            
            // Update conversation preview in sidebar
            const conversationItem = document.querySelector(`.conversation-item[data-id="${currentConversationId}"]`);
            if (conversationItem) {
              const previewEl = conversationItem.querySelector('.conversation-preview');
              if (previewEl) {
                previewEl.textContent = `Assistant: ${data.response.substring(0, 30)}${data.response.length > 30 ? '...' : ''}`;
              }
            }
            
            // Update conversation title if it's the first message
            if (conversations[currentConversationId].messages.length === 2) { // User + assistant
              // Update the title with the first user message
              const titleEl = conversationItem?.querySelector('.conversation-title');
              if (titleEl) {
                titleEl.textContent = query.substring(0, 30) + (query.length > 30 ? '...' : '');
              }
              
              // Also update the current conversation name
              currentConversationName.textContent = query.substring(0, 30) + (query.length > 30 ? '...' : '');
            }
          }
          
          // Show feedback options
          showFeedbackOptions(query, data.response);
        } else {
          const errorData = await response.json();
          console.error('Query error:', errorData);
          addMessage('assistant', `Error: ${errorData.error || 'Failed to get response'}`);
        }
      } catch (error) {
        console.error('Error sending query:', error);
        addMessage('assistant', 'Sorry, an error occurred while processing your request.');
      }
    }
    
    // Add a message to the chat
function addMessage(role, content, context = null) {
  const messageEl = document.createElement('div');
  messageEl.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
  messageEl.innerHTML = content.replace(/\n/g, '<br>');

  chatMessages.appendChild(messageEl);

  // If RAG and assistant with context — add separate context section below
  if (
    role === 'assistant' &&
    modelSelect.value === 'rag' &&
    context &&
    context.trim() !== ''
  ) {
    const wrapper = document.createElement('div');
    wrapper.className = 'rag-context-wrapper';

    const toggleBtn = document.createElement('button');
    toggleBtn.textContent = 'Show Context ⬇️';
    toggleBtn.className = 'toggle-context-btn';

    const contextBox = document.createElement('div');
    contextBox.className = 'context-box';
    contextBox.innerHTML = context.replace(/\n/g, '<br>');
    contextBox.style.display = 'none';

    toggleBtn.addEventListener('click', () => {
      const isHidden = contextBox.style.display === 'none';
      contextBox.style.display = isHidden ? 'block' : 'none';
      toggleBtn.textContent = isHidden ? 'Hide Context ⬆️' : 'Show Context ⬇️';
    });

    wrapper.appendChild(toggleBtn);
    wrapper.appendChild(contextBox);
    chatMessages.appendChild(wrapper);
  }

  // Scroll to bottom
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
    
    // Upload document (admin only)
    async function uploadDocument(form) {
      try {
        const formData = new FormData(form);
        
        const response = await fetch('/upload', {
          method: 'POST',
          body: formData
        });
        
        if (response.ok) {
          const data = await response.json();
          alert(`File ${data.filename} uploaded successfully!`);
          document.getElementById('uploadModal').style.display = 'none';
        } else {
          const errorData = await response.json();
          alert(`Upload failed: ${errorData.message}`);
        }
      } catch (error) {
        console.error('Error uploading document:', error);
        alert('An error occurred while uploading the document.');
      }
    }
    
    // Show feedback options
    function showFeedbackOptions(query, response) {
      const feedbackSection = document.getElementById('textFeedbackSection');
      const feedbackThanks = document.getElementById('feedbackThanks');
      
      feedbackSection.style.display = 'block';
      feedbackThanks.style.display = 'none';
      
      // Store query and response for feedback
      feedbackSection.dataset.query = query;
      feedbackSection.dataset.response = response;
    }
    
    // Send feedback comment
    window.sendFeedbackComment = function() {
      const feedbackSection = document.getElementById('textFeedbackSection');
      const feedbackThanks = document.getElementById('feedbackThanks');
      const feedbackTextarea = document.getElementById(`${userType}-feedback-comment`);
      
      const feedbackText = feedbackTextarea.value.trim();
      if (!feedbackText) return;
      
      const query = feedbackSection.dataset.query;
      const response = feedbackSection.dataset.response;
      
      sendFeedback(query, response, feedbackText);
      
      // Hide feedback section, show thank you
      feedbackSection.style.display = 'none';
      feedbackThanks.style.display = 'block';
      feedbackTextarea.value = '';
      
      // Hide thank you after 3 seconds
      setTimeout(() => {
        feedbackThanks.style.display = 'none';
      }, 3000);
    };
    
    // Send feedback to server
    async function sendFeedback(query, response, feedback) {
      try {
        const feedbackData = {
          query: query,
          response: response,
          feedback: feedback,
          email: userEmail,
          timestamp: new Date().toISOString(),
          userType: userType,
          model_type: modelSelect.value,
          conversation_id: currentConversationId
        };
        
        const serverResponse = await fetch('/submit_feedback', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(feedbackData)
        });
        
        if (!serverResponse.ok) {
          console.error('Failed to submit feedback');
        }
      } catch (error) {
        console.error('Error sending feedback:', error);
      }
    }
  });