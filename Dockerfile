FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8501
# ENV OPENAI_API_KEY=your_api_key_here

CMD ["streamlit", "run", "app.py"]
