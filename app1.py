
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load data
news_df = pd.read_csv('D:/Projects/Fake News Prediction/train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']

# Define lemmatization function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  # Avoid reloading stopwords every time

def lemmatize(content):
    lemmatized_content = re.sub('[^a-zA-Z]', ' ', content)
    lemmatized_content = lemmatized_content.lower().split()
    lemmatized_content = [lemmatizer.lemmatize(word) for word in lemmatized_content if word not in stop_words]
    return ' '.join(lemmatized_content)

# Apply lemmatization function to content column
news_df['content'] = news_df['content'].apply(lemmatize)

# Vectorize data with bigrams (n-grams: unigrams + bigrams)
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer(ngram_range=(1, 2))  # Using both unigrams and bigrams
vector.fit(X)
X = vector.transform(X)

# Check dataset balance
print("Class distribution in the dataset:\n", news_df['label'].value_counts())

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model with class balancing
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, Y_train)

# Evaluate the model on test data
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(Y_test, y_pred))

# Create Dash app
app = dash.Dash(__name__)

# Update layout to hide pie chart initially
app.layout = html.Div([
    html.H1('Fake News Detector', style={
        'textAlign': 'center',
        'color': '#FFFFFF',
        'backgroundColor': '#007BFF',
        'padding': '20px',
        'font-family': 'Arial'
    }),
    html.Div([
        html.P('Please enter the news article in the format: Author Name _ News Article Title _ News Content (if possible)', style={
            'textAlign': 'center',
            'color': '#555',
            'font-family': 'Arial',
            'margin-bottom': '10px'
        }),
        dcc.Input(id='input-text', type='text', placeholder='Enter content ...', style={
            'width': '70%',
            'padding': '10px',
            'font-size': '18px',
            'border': '2px solid #007BFF',
            'border-radius': '5px',
            'margin-right': '10px'
        }),
        html.Button('Check', id='submit-btn', n_clicks=0, style={
            'padding': '10px 20px',
            'font-size': '18px',
            'background-color': '#007BFF',
            'color': '#FFFFFF',
            'border': 'none',
            'border-radius': '5px',
            'cursor': 'pointer'
        }),
    ], style={'textAlign': 'center', 'margin-top': '20px'}),

    html.Div(id='output-result', style={
        'margin-top': '30px',
        'textAlign': 'center',
        'font-size': '22px',
        'font-family': 'Arial',
    }),

    # Hide the pie chart initially by setting display to 'none'
    dcc.Graph(id='probability-pie-chart', style={'display': 'none'}),  # Initially hidden
], style={
    'backgroundColor': '#F7F7F7',
    'height': '100vh',
    'padding-top': '50px'
})


# Define the prediction function with probability output
def predict_fake_news(input_text):
   # print("Original Input:", input_text)
    lemmatized_input = lemmatize(input_text)
   # print("Lemmatized Input:", lemmatized_input)
    input_data = vector.transform([lemmatized_input])
    probability = model.predict_proba(input_data)[0]  # Get probabilities for both classes
    prediction = model.predict(input_data)[0]
    return prediction, probability
# Update the callback to show the pie chart conditionally
@app.callback(
    [Output('output-result', 'children'),
     Output('probability-pie-chart', 'figure'),
     Output('probability-pie-chart', 'style')],  # Adding a new Output for style
    [Input('submit-btn', 'n_clicks'),
     Input('input-text', 'value')]
)
def update_output(n_clicks, input_text):
    if n_clicks > 0 and input_text:  # Ensure there's input and the button has been clicked
        pred, prob = predict_fake_news(input_text)
        fake_prob = prob[1]  # Probability of being fake
        real_prob = prob[0]  # Probability of being real

        # Display text output
        if pred == 1:
            result_text = html.Div([
                html.Div(f'The News is Fake (Probability: {fake_prob*100:.2f}%)', style={'color': 'red'}),
                html.Div(f'Real News Probability: {real_prob*100:.2f}%', style={'color': 'green'})
            ])
        else:
            result_text = html.Div([
                html.Div(f'The News is Real (Probability: {real_prob*100:.2f}%)', style={'color': 'green'}),
                html.Div(f'Fake News Probability: {fake_prob*100:.2f}%', style={'color': 'red'})
            ])

        # Create pie chart
        pie_chart = go.Figure(data=[go.Pie(labels=['Real', 'Fake'],
                                           values=[real_prob, fake_prob],
                                           hole=0.4,
                                           marker=dict(colors=['green', 'red']))])  # Adding a hole in the center to make it a donut chart

        pie_chart.update_layout(
            title_text='Probability Distribution of News',
            annotations=[dict(text=f'{max(real_prob, fake_prob) * 100:.2f}%',
                              x=0.5, y=0.5,
                              font_size=20, showarrow=False)],
            showlegend=True
        )

        # Show the pie chart by setting display to 'block'
        return result_text, pie_chart, {'display': 'block'}

    # Return default values before user input
    return '', go.Figure(), {'display': 'none'}  # Hide the pie chart initially

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

