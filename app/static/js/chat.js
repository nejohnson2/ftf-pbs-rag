// Chat UI helpers

function scrollToBottom() {
  const messages = document.getElementById('messages');
  if (messages) {
    messages.scrollTop = messages.scrollHeight;
  }
}

function setLoading(isLoading) {
  const btn = document.getElementById('submit-btn');
  const input = document.getElementById('query-input');
  if (btn) btn.disabled = isLoading;
  if (input) input.disabled = isLoading;

  // Show/hide a typing indicator
  const existing = document.getElementById('typing-indicator');
  if (isLoading && !existing) {
    const messages = document.getElementById('messages');
    const indicator = document.createElement('div');
    indicator.id = 'typing-indicator';
    indicator.className = 'message assistant-message';
    indicator.innerHTML = '<div class="message-body"><span class="typing-dots">Thinking…</span></div>';
    messages.appendChild(indicator);
    scrollToBottom();
  } else if (!isLoading && existing) {
    existing.remove();
  }
}

// Ctrl+Enter submits the form
document.addEventListener('DOMContentLoaded', () => {
  const textarea = document.getElementById('query-input');
  if (textarea) {
    textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.ctrlKey) {
        e.preventDefault();
        const form = document.getElementById('chat-form');
        if (form) htmx.trigger(form, 'submit');
      }
    });
  }

  // Auto-scroll on new content from HTMX
  document.body.addEventListener('htmx:afterSwap', () => {
    scrollToBottom();
  });
});
