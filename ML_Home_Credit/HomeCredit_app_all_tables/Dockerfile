    # execute with:  docker build -t payment-app-only:latest .
 
    
    FROM python:3.9-slim 
    #: Specifies the container’s base image as Python version 3.9 with the slim variant, 
    # which is a smaller image compared to the regular Python image
    
    WORKDIR /app         
    #: This instruction sets the working directory inside the container to /app
    
    COPY requirements.txt requirements.txt  
    #: This instruction copies the requirements.txt file from the host machine to the 
    # containers app directory

    RUN pip install --upgrade pip 
    RUN pip install -r requirements.txt  
    #: This instruction upgrades pip and installs the required packages specified in the 
    # requirements.txt file using pip

    COPY ["defaultrisk_model_all_tables.pickle", "app.py", "./"] .    
    #: This instruction copies the trained CatBoost model, and the app.py file, 
    # which contains the Python code for the application, from the host machine to the
    # containers /app directory

    EXPOSE 8080
    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]   
    #: This instruction specifies the command that should be run when the container starts. 
    # It runs the uvicorn server with the app module as the main application at URL 0.0.0.0:8080