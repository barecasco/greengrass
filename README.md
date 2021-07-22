## Greengrass
Machine learning use case using classification algorithm. 
The app works as a webservice that retrieve data as json and return the prediction.

Built using Flask.

### Run on Docker
To build docker image, set your directory to where these files are located. Then build using:

```docker build -t greengrass .```

Run the image using Docker Desktop by exposing port 5000, or run in detached mode by using docker command:

```docker run -dp 5000:5000 greengrass```

### Testing the app
I've created `post_test.py` file that will send requests to localhost:5000. Run the file using python after the docker image is run.

### Analysis
The thoughts on modeling are writen using comment in `modeling.py` file.
The functions in `eda_utils.py` are used in exploratory data analysis and data cleaning. The thought process on cleaning is also written in comment inside the function.
The Flask app is built using `app.py` file, where saved model is utilized to predict incoming json data.
