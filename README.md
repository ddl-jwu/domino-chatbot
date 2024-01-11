# Domino chatbot

# Environment Defintion

Base image: `Domino Standard Environment Py3.9 R4.3` (in Domino 5.10)

Additional dockerfile instructions:
```
RUN pip install streamlit pypdf pinecone-client ipywidgets langchain
```

# Local Testing

Create a project and mount this git repository instead of using the default Domino File System.

To test in workspace:

1. Change ports in `app.sh` to `8887`
2. Run `./app.sh`
3. Go to `https://{domino-url}/{username}/{domino-project-name}/notebookSession/{runId}/proxy/8887/`
