import fetch from 'node-fetch';

const CHATBOT_API_URL = 'http://35.9.219.76:5000/api/generate';

async function testChatbot() {
  try {
    const response = await fetch(CHATBOT_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'deepseek-r1:70b',
        prompt: 'What is the capital of France?',
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    console.log('Chatbot Response:', result);
  } catch (error) {
    console.error('Error:', error);
  }
}

testChatbot();
