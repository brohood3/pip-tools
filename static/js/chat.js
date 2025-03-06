document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const messageContainer = document.getElementById('messageContainer');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const typingIndicator = document.getElementById('typingIndicator');
    const statusText = document.getElementById('statusText');
    const tokenCounter = document.getElementById('tokenCounter');
    const toggleToolsBtn = document.getElementById('toggleTools');
    const modelSelect = document.getElementById('modelSelect');
    const clearChatBtn = document.getElementById('clearChat');
    const logoutBtn = document.getElementById('logoutBtn');
    const darkModeToggle = document.getElementById('darkModeToggle');

    // Initialize dark mode from localStorage
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    if (isDarkMode) {
        document.body.classList.add('dark-mode');
        darkModeToggle.innerHTML = '<i class="fa-solid fa-sun"></i><span>Light</span>';
    }

    // Socket.io connection
    const socket = io();

    // State
    let useTools = true;
    let conversation = [];
    let isThinking = false;

    // Initialize highlight.js
    hljs.highlightAll();

    // Set up markdown renderer with code highlighting
    const renderer = new marked.Renderer();
    renderer.code = (code, language) => {
        const validLanguage = hljs.getLanguage(language) ? language : 'plaintext';
        const highlightedCode = hljs.highlight(validLanguage, code).value;
        return `<pre><code class="hljs ${validLanguage}">${highlightedCode}</code></pre>`;
    };

    marked.setOptions({
        renderer,
        highlight: (code, lang) => {
            const language = hljs.getLanguage(lang) ? lang : 'plaintext';
            return hljs.highlight(code, { language }).value;
        },
        breaks: true
    });

    // Event Listeners
    userInput.addEventListener('keydown', handleInputKeydown);
    sendButton.addEventListener('click', sendMessage);
    toggleToolsBtn.addEventListener('click', toggleTools);
    modelSelect.addEventListener('change', changeModel);
    clearChatBtn.addEventListener('click', clearChat);
    logoutBtn.addEventListener('click', logout);
    darkModeToggle.addEventListener('click', toggleDarkMode);

    // Auto-resize textarea as user types
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
    });

    // WebSocket event handlers
    socket.on('connect', () => {
        setStatus('Connected');
    });

    socket.on('disconnect', () => {
        setStatus('Disconnected');
    });

    socket.on('typing_indicator', (data) => {
        if (data.status === 'typing') {
            showTypingIndicator();
        } else {
            hideTypingIndicator();
        }
    });

    socket.on('status_update', (data) => {
        setStatus(data.status);
    });

    // Functions
    function sendMessage() {
        const message = userInput.value.trim();
        if (message && !isThinking) {
            // Disable input while thinking
            isThinking = true;
            userInput.disabled = true;
            sendButton.disabled = true;

            // Add user message to UI
            addMessageToUI('user', message);
            
            // Clear input
            userInput.value = '';
            userInput.style.height = 'auto';

            // Show typing indicator
            showTypingIndicator();
            setStatus('Thinking...');

            // Send to server
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    use_tools: useTools
                }),
            })
            .then(response => response.json())
            .then(data => {
                // Hide typing indicator
                hideTypingIndicator();
                
                // Add assistant message to UI
                addMessageToUI('assistant', data.text, data);
                
                // Update token counter
                if (data.usage) {
                    updateTokenCounter(data.usage.total_tokens);
                }

                // Re-enable input
                isThinking = false;
                userInput.disabled = false;
                sendButton.disabled = false;
                userInput.focus();

                // Set status
                setStatus('Ready');
            })
            .catch(error => {
                console.error('Error:', error);
                hideTypingIndicator();
                setStatus('Error: ' + error.message);
                isThinking = false;
                userInput.disabled = false;
                sendButton.disabled = false;
            });

            // Save to conversation history
            conversation.push({
                role: 'user',
                content: message
            });
        }
    }

    function addMessageToUI(role, content, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        // Add avatar based on role
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        
        if (role === 'user') {
            avatar.innerHTML = '<i class="fa-solid fa-user"></i>';
        } else {
            avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';
        }
        
        messageDiv.appendChild(avatar);

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Use marked.js to render markdown
        messageContent.innerHTML = marked.parse(content);
        
        messageDiv.appendChild(messageContent);

        // Add chart image if available
        if (metadata.chart) {
            const chartImg = document.createElement('img');
            chartImg.src = `data:image/png;base64,${metadata.chart}`;
            chartImg.alt = 'Technical Analysis Chart';
            chartImg.style.maxWidth = '100%';
            chartImg.style.borderRadius = '8px';
            chartImg.style.marginTop = '10px';
            messageContent.appendChild(chartImg);
        }

        // Add metadata if available
        if (role === 'assistant' && Object.keys(metadata).length > 0) {
            const metadataDiv = document.createElement('div');
            metadataDiv.className = 'message-metadata';

            // Add timestamp
            const timeSpan = document.createElement('span');
            timeSpan.className = 'message-time';
            const timestamp = metadata.timestamp ? new Date(metadata.timestamp) : new Date();
            timeSpan.textContent = formatTime(timestamp);
            metadataDiv.appendChild(timeSpan);

            // Add tool badge if tool was used
            if (metadata.tool_used) {
                const toolSpan = document.createElement('span');
                toolSpan.className = 'tool-badge';
                toolSpan.innerHTML = `<i class="fa-solid fa-wrench"></i> ${metadata.tool_used}`;
                metadataDiv.appendChild(toolSpan);
            }

            messageDiv.appendChild(metadataDiv);
        }

        messageContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        messageContainer.scrollTop = messageContainer.scrollHeight;
        
        // Apply syntax highlighting to code blocks
        messageDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightBlock(block);
        });

        // Save to conversation history
        if (role !== 'system') {
            conversation.push({
                role: role,
                content: content
            });
        }
    }

    function handleInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    }

    function showTypingIndicator() {
        typingIndicator.classList.remove('hide');
    }

    function hideTypingIndicator() {
        typingIndicator.classList.add('hide');
    }

    function setStatus(status) {
        statusText.textContent = status;
    }

    function updateTokenCounter(count) {
        tokenCounter.textContent = `Tokens: ${count}`;
    }

    function toggleTools() {
        useTools = !useTools;
        toggleToolsBtn.querySelector('span').textContent = `Tools: ${useTools ? 'On' : 'Off'}`;
    }

    function changeModel() {
        const model = modelSelect.value;
        fetch('/api/models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ model }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                setStatus(`Model changed to ${model}`);
            }
        })
        .catch(error => {
            console.error('Error changing model:', error);
            setStatus('Error changing model');
        });
    }

    function clearChat() {
        // Clear UI
        while (messageContainer.children.length > 1) {
            messageContainer.removeChild(messageContainer.lastChild);
        }
        
        // Reset conversation
        conversation = [];
        
        // Reset token counter
        fetch('/api/reset', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateTokenCounter(0);
                setStatus('Chat cleared');
            }
        })
        .catch(error => {
            console.error('Error resetting usage:', error);
        });
    }

    function formatTime(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function logout() {
        // Show confirmation dialog
        if (confirm('Are you sure you want to logout?')) {
            fetch('/auth/logout', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => {
                if (response.ok) {
                    // Redirect to login page
                    window.location.href = '/login';
                } else {
                    console.error('Logout failed');
                    alert('Failed to logout. Please try again.');
                }
            })
            .catch(error => {
                console.error('Logout error:', error);
                alert('An error occurred during logout.');
            });
        }
    }

    function toggleDarkMode() {
        const isDarkMode = document.body.classList.toggle('dark-mode');
        
        // Update button icon and text
        if (isDarkMode) {
            darkModeToggle.innerHTML = '<i class="fa-solid fa-sun"></i><span>Light</span>';
        } else {
            darkModeToggle.innerHTML = '<i class="fa-solid fa-moon"></i><span>Dark</span>';
        }
        
        // Save preference to localStorage
        localStorage.setItem('darkMode', isDarkMode);
    }

    // Initialize
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            // Set current model in dropdown
            modelSelect.value = data.current;
            
            // Fetch token usage
            return fetch('/api/usage');
        })
        .then(response => response.json())
        .then(data => {
            // Update token counter
            updateTokenCounter(data.total_tokens);
        })
        .catch(error => {
            console.error('Error initializing:', error);
        });
}); 