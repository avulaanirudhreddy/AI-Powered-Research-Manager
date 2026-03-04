# Founder's Community Dev Trial

Built an AI-powered Research Assistant using **LangChain** and **OpenAI's GPT model**. The application enables users to efficiently interact with research papers by generating summaries, extracting citations and references, and answering specific queries related to the paper’s content.

## Features

- **Research Paper Summary**: Automatically generates a concise summary of the input research paper.
- **Citations and References Extraction**: Extracts a list of citations and references from the research paper.
- **Query Response**: Provides answers to user queries based on the content of the research paper.

## Technologies Used

- **LangChain**: A framework for building applications using language models.
- **OpenAI GPT**: Utilized for natural language understanding and generation.
- **Streamlit**: A framework to quickly build and deploy the web application.
- **Python**: Core programming language for development.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/avulaanirudhreddy/founders-community-dev-trial
   cd founders-community-dev-trial
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the application:
   ```bash
   streamlit run app.py

Visit http://localhost:8501 in your browser to access the app.

## How It Works

<ul>
    <li>Upload a research paper: Users upload a PDF of the research paper they want to work with.</li>
    <li>Generate summary: The model processes the text and generates a concise summary.</li>
    <li>Extract citations: The system extracts references and citations from the document.</li>
    <li>Answer queries: The model can answer any specific queries related to the content of the paper.</li>
</ul>

## Contributing

<ol>
    <li>Fork the repository.</li>
    <li>Create your branch (git checkout -b feature-name).</li>
    <li>Commit your changes (git commit -am 'Add new feature').</li>
    <li>Push to the branch (git push origin feature-name).</li>
    <li>Create a new Pull Request.</li>
</ol>


## Acknowledgements

<ul>
    <li>OpenAI for providing powerful language models.</li>
    <li>LangChain for simplifying the process of building LLM-powered applications.</li>
    <li>Streamlit for easy deployment and sharing of web applications.</li>
</ul>
