# Project Intro
I build this project to showcase the functionality and versability of my python package -> [llm-agent-toolkit](https://github.com/JonahWhaler/llm-agent-toolkit).

## Features [ 🧠 | 🔨 | :rocket: ]
| Feature                  | Status   | Notes                                                                                                                |
| ------------------------ | -------- | -------------------------------------------------------------------------------------------------------------------- |
| Interect with text       | :rocket: |                                                                                                                      |
| Interect with voice      | :rocket: | Transform audio to transcript                                                                                        |
| Photo Upload             | :rocket: | Interpret the uploaded images                                                                                        |
| Audio Upload             | :rocket: | Transform audio to transcript                                                                                        |
| File Upload              | :rocket: | Extract text content from file including embedded images                                                             |
| Chat with memory         | :rocket: | Maintain a limited number of conversation                                                                            |
| User Preference          | :rocket: | Maintain user's preference and insert it to every call to offer responses tailored to user                           |
| Character Routing        | :rocket: | Let llm pick the character/avatar relevant to the user's prompt                                                      |
| Web Search               | :rocket: | Implement as a tool, allow llm to search the internet through DuckDuckGo API. Retrieve the page content if available |
| Smart Web Search         | :rocket: | Enhanced Web Search, have another llm to pick relevant portion from the page content                                 |
| Ollama Support           | :rocket: | v0.4.4 [Text, Vision, Embedding]                                                                                     |
| OpenAI Support           | :rocket: | v1.58.0 [Text, Vision, Embedding, Transcript]                                                                        |
| DeepSeek Support         | :rocket: | v1.58.0 [Text]                                                                                                       |
| Gemini Support           | :rocket: | v1.0.0 [Text, Vision]                                                                                                |
| Local Whisper            | :rocket: | openai-whisper==20240930                                                                                             |
| Rate Limiting            | :rocket: | Basic rate limiting using [pygrl](https://github.com/JonahWhaler/rate-limiter)                                       |
| Personal Vector Memory   | :rocket: | Maintain file content embeddings in a Vector Memory [Add \| Update \| Query \|Delete] to support RAG                 |
| Voice Response           | :brain:  | Reply to user with voices                                                                                            |
| Anthropic Support        | :brain:  |                                                                                                                      |
| Image Generation         | :brain:  |                                                                                                                      |
| User Access Control      | :rocket: | Allow system admin to control user's access [Allow \| Suspend]                                                       |
| Broadcast                | :brain:  | Allow system admin to push updates to users                                                                          |
| Content Moderation       | :brain:  | Validate whether user\'s input and LLM\'s responses are compliance to the ethical boundary                           |
| File Response            | :brain:  | Response to user with file with shareable link                                                                       |
| Reasoning Models         | :rocket:  | Integrate with reasoning models                                                                                      |
| Group Support            | :brain:  | Let this bot live in group chat                                                                                      |
| Edit Message             | :brain:  | Response to message edit                                                                                             |
| Incoming Forward Message | :brain:  | Response to incoming message                                                                                         |
| Retry Generation         | :brain:  | Trigger the LLM to re-run the last execution                                                                         |

---


## Constaints

- Telegram's Upload Limit: [File size should not exceed 20MB](https://core.telegram.org/bots/faq#handling-media)
- Telegram's Rate Limiting: [Broadcasting to Users](https://core.telegram.org/bots/faq#handling-media)
- Telegram's Character Limit for Messages: [Issue](https://bugs.telegram.org/c/1423)
- Telegram's Character Limit for Media Captions: [Issue](https://bugs.telegram.org/c/1022)
- Developer's Development Environment: I only have machine with CPU. I won't be able to test any GPU related features.

## Out-of-Scope Features

- Monetization: Token management
